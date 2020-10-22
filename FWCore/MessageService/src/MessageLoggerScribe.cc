// ----------------------------------------------------------------------
//
// MessageLoggerScribe.cc
//
// ----------------------------------------------------------------------

#include "FWCore/MessageService/src/MessageLoggerScribe.h"
#include "FWCore/MessageService/src/ELadministrator.h"
#include "FWCore/MessageService/src/ELoutput.h"
#include "FWCore/MessageService/src/ELstatistics.h"
#include "FWCore/MessageService/src/ThreadQueue.h"

#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageService/src/ConfigurationHandshake.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <csignal>

using std::cerr;

namespace edm {
  namespace service {

    MessageLoggerScribe::MessageLoggerScribe(std::shared_ptr<ThreadQueue> queue)
        : admin_p(new ELadministrator()),
          early_dest(admin_p->attach(std::make_shared<ELoutput>(std::cerr, false))),
          file_ps(),
          job_pset_p(),
          clean_slate_configuration(true),
          active(true),
          singleThread(queue.get() == nullptr),
          done(false),
          purge_mode(false),
          count(false),
          m_queue(queue) {}

    MessageLoggerScribe::~MessageLoggerScribe() { admin_p->finish(); }

    void MessageLoggerScribe::run() {
      MessageLoggerQ::OpCode opcode;
      void* operand;

      MessageDrop::instance()->messageLoggerScribeIsRunning = MLSCRIBE_RUNNING_INDICATOR;
      //  std::cerr << "MessageLoggerScribe::run(): \n";
      //  std::cerr << "messageLoggerScribeIsRunning = "
      //      << (int)MessageDrop::instance()->messageLoggerScribeIsRunning << "\n";

      do {
        m_queue->consume(opcode, operand);  // grab next work item from Q
        runCommand(opcode, operand);
      } while (!done);

    }  // MessageLoggerScribe::run()

    void MessageLoggerScribe::runCommand(MessageLoggerQ::OpCode opcode, void* operand) {
      switch (opcode) {  // interpret the work item
        default: {
          assert(false);  // can't happen (we certainly hope!)
          break;
        }
        case MessageLoggerQ::END_THREAD: {
          assert(operand == nullptr);
          done = true;
          MessageDrop::instance()->messageLoggerScribeIsRunning = (unsigned char)-1;
          break;
        }
        case MessageLoggerQ::LOG_A_MESSAGE: {
          ErrorObj* errorobj_p = static_cast<ErrorObj*>(operand);
          try {
            if (active && !purge_mode)
              log(errorobj_p);
          } catch (cms::Exception& e) {
            ++count;
            std::cerr << "MessageLoggerScribe caught " << count << " cms::Exceptions, text = \n" << e.what() << "\n";

            if (count > 25) {
              cerr << "MessageLogger will no longer be processing "
                   << "messages due to errors (entering purge mode).\n";
              purge_mode = true;
            }
          } catch (...) {
            std::cerr << "MessageLoggerScribe caught an unknown exception and "
                      << "will no longer be processing "
                      << "messages. (entering purge mode)\n";
            purge_mode = true;
          }
          delete errorobj_p;  // dispose of the message text
          break;
        }
        case MessageLoggerQ::CONFIGURE: {
          if (singleThread) {
            job_pset_p =
                std::shared_ptr<PSet>(static_cast<PSet*>(operand));  // propagate_const<T> has no reset() function
            configure_errorlog();
            break;
          } else {
            ConfigurationHandshake* h_p = static_cast<ConfigurationHandshake*>(operand);
            job_pset_p =
                std::shared_ptr<PSet>(static_cast<PSet*>(h_p->p));  // propagate_const<T> has no reset() function
            std::lock_guard<std::mutex> sl(h_p->m);                 // get lock
            try {
              configure_errorlog();
            } catch (edm::Exception& e) {
              Place_for_passing_exception_ptr epp = h_p->epp;
              if (!(*epp)) {
                *epp = std::make_shared<edm::Exception>(e);
              } else {
                Pointer_to_new_exception_on_heap ep = *epp;
                (*ep) << "\n and another exception: \n" << e.what();
              }
            }
            // Note - since the configuring code has not made a new copy of the
            // job parameter set, we must not delete job_pset_p (in contrast to
            // the case for errorobj_p).  On the other hand, if we instantiate
            // a new edm::Exception pointed to by *epp, it is the responsibility
            // of the MessageLoggerQ to delete it.
            h_p->c.notify_all();  // Signal to MessageLoggerQ that we are done
            // finally, release the scoped lock by letting it go out of scope
            break;
          }
        }
        case MessageLoggerQ::SUMMARIZE: {
          assert(operand == nullptr);
          try {
            triggerStatisticsSummaries();
          } catch (cms::Exception& e) {
            std::cerr << "MessageLoggerScribe caught exception "
                      << "during summarize:\n"
                      << e.what() << "\n";
          } catch (...) {
            std::cerr << "MessageLoggerScribe caught unkonwn exception type "
                      << "during summarize. (Ignored)\n";
          }
          break;
        }
        case MessageLoggerQ::JOBMODE: {
          std::string* jobMode_p = static_cast<std::string*>(operand);
          JobMode jm = MessageLoggerDefaults::mode(*jobMode_p);
          messageLoggerDefaults = value_ptr<MessageLoggerDefaults>(new MessageLoggerDefaults(jm));
          // Note - since messageLoggerDefaults is a value_ptr,
          //        there is no concern about deleting here.
          delete jobMode_p;  // dispose of the message text
                             // which will have been new-ed
                             // in MessageLogger.cc (service version)
          break;
        }
        case MessageLoggerQ::SHUT_UP: {
          assert(operand == nullptr);
          active = false;
          break;
        }
        case MessageLoggerQ::FLUSH_LOG_Q: {
          if (singleThread)
            return;
          ConfigurationHandshake* h_p = static_cast<ConfigurationHandshake*>(operand);
          job_pset_p = std::shared_ptr<PSet>(static_cast<PSet*>(h_p->p));  // propagate_const<T> has no reset() function
          std::lock_guard<std::mutex> sl(h_p->m);                          // get lock
          h_p->c.notify_all();                                             // Signal to MessageLoggerQ that we are done
          // finally, release the scoped lock by letting it go out of scope
          break;
        }
        case MessageLoggerQ::GROUP_STATS: {
          std::string* cat_p = static_cast<std::string*>(operand);
          ELstatistics::noteGroupedCategory(*cat_p);
          delete cat_p;  // dispose of the message text
          break;
        }
        case MessageLoggerQ::FJR_SUMMARY: {
          if (singleThread) {
            std::map<std::string, double>* smp = static_cast<std::map<std::string, double>*>(operand);
            triggerFJRmessageSummary(*smp);
            break;
          } else {
            ConfigurationHandshake* h_p = static_cast<ConfigurationHandshake*>(operand);
            std::lock_guard<std::mutex> sl(h_p->m);  // get lock
            std::map<std::string, double>* smp = static_cast<std::map<std::string, double>*>(h_p->p);
            triggerFJRmessageSummary(*smp);
            h_p->c.notify_all();  // Signal to MessageLoggerQ that we are done
            // finally, release the scoped lock by letting it go out of scope
            break;
          }
        }
      }  // switch

    }  // MessageLoggerScribe::runCommand(opcode, operand)

    void MessageLoggerScribe::log(ErrorObj* errorobj_p) {
      std::vector<std::string> categories;
      parseCategories(errorobj_p->xid().id, categories);
      for (unsigned int icat = 0; icat < categories.size(); ++icat) {
        errorobj_p->setID(categories[icat]);
        admin_p->log(*errorobj_p);  // route the message text
      }
    }

    void MessageLoggerScribe::configure_errorlog() {
      vString empty_vString;
      String empty_String;
      PSet empty_PSet;

      // The following is present to test pre-configuration message handling:
      String preconfiguration_message =
          getAparameter<String>(*job_pset_p, "generate_preconfiguration_message", empty_String);
      if (preconfiguration_message != empty_String) {
        // To test a preconfiguration message without first going thru the
        // configuration we are about to do, we issue the message (so it sits
        // on the queue), then copy the processing that the LOG_A_MESSAGE case
        // does.  We suppress the timestamp to allow for automated unit testing.
        early_dest->suppressTime();
        LogError("preconfiguration") << preconfiguration_message;
        if (!singleThread) {
          MessageLoggerQ::OpCode opcode;
          void* operand;
          m_queue->consume(opcode, operand);  // grab next work item from Q
          assert(opcode == MessageLoggerQ::LOG_A_MESSAGE);
          ErrorObj* errorobj_p = static_cast<ErrorObj*>(operand);
          log(errorobj_p);
          delete errorobj_p;  // dispose of the message text
        }
      }

      if (!stream_ps.empty()) {
        LogWarning("multiLogConfig") << "The message logger has been configured multiple times";
        clean_slate_configuration = false;
      }
      configure_ordinary_destinations();
      configure_statistics();
    }  // MessageLoggerScribe::configure_errorlog()

    void MessageLoggerScribe::configure_dest(std::shared_ptr<ELdestination> dest_ctrl, String const& filename) {
      static const int NO_VALUE_SET = -45654;
      vString empty_vString;
      PSet empty_PSet;
      String empty_String;

      // Defaults:
      const std::string COMMON_DEFAULT_THRESHOLD = "INFO";
      const int COMMON_DEFAULT_LIMIT = NO_VALUE_SET;
      const int COMMON_DEFAULT_INTERVAL = NO_VALUE_SET;
      const int COMMON_DEFAULT_TIMESPAN = NO_VALUE_SET;

      vString const severities = {{"WARNING", "INFO", "FWKINFO", "ERROR", "DEBUG"}};

      // grab list of categories
      vString categories = getAparameter<vString>(*job_pset_p, "categories", empty_vString);

      // grab list of hardwired categories (hardcats) -- these are to be added
      // to the list of categories
      {
        std::vector<std::string> hardcats = messageLoggerDefaults->categories;
        // combine the lists, not caring about possible duplicates (for now)
        copy_all(hardcats, std::back_inserter(categories));
      }  // no longer need hardcats

      // grab default threshold common to all destinations
      String default_threshold = getAparameter<String>(*job_pset_p, "threshold", empty_String);

      // grab default limit/interval/timespan common to all destinations/categories:
      PSet default_pset = getAparameter<PSet>(*job_pset_p, "default", empty_PSet);
      int default_limit = getAparameter<int>(default_pset, "limit", COMMON_DEFAULT_LIMIT);
      int default_interval = getAparameter<int>(default_pset, "reportEvery", COMMON_DEFAULT_INTERVAL);
      int default_timespan = getAparameter<int>(default_pset, "timespan", COMMON_DEFAULT_TIMESPAN);
      String default_pset_threshold = getAparameter<String>(default_pset, "threshold", default_threshold);

      // grab all of this destination's parameters:
      PSet dest_pset = getAparameter<PSet>(*job_pset_p, filename, empty_PSet);

      // See if this is just a placeholder
      bool is_placeholder = getAparameter<bool>(dest_pset, "placeholder", false);
      if (is_placeholder)
        return;

      // grab this destination's default limit/interval/timespan:
      PSet dest_default_pset = getAparameter<PSet>(dest_pset, "default", empty_PSet);
      int dest_default_limit = getAparameter<int>(dest_default_pset, "limit", default_limit);
      int dest_default_interval = getAparameter<int>(dest_default_pset, "reportEvery", default_interval);
      int dest_default_timespan = getAparameter<int>(dest_default_pset, "timespan", default_timespan);
      if (dest_default_limit != NO_VALUE_SET) {
        if (dest_default_limit < 0)
          dest_default_limit = 2000000000;
        dest_ctrl->setLimit("*", dest_default_limit);
      }
      if (dest_default_interval != NO_VALUE_SET) {
        dest_ctrl->setInterval("*", dest_default_interval);
      }
      if (dest_default_timespan != NO_VALUE_SET) {
        if (dest_default_timespan < 0)
          dest_default_timespan = 2000000000;
        dest_ctrl->setTimespan("*", dest_default_timespan);
      }

      // establish this destination's threshold:
      String dest_threshold = getAparameter<String>(dest_pset, "threshold", default_threshold);
      if (dest_threshold == empty_String) {
        dest_threshold = default_threshold;
      }
      if (dest_threshold == empty_String) {
        dest_threshold = default_pset_threshold;
      }
      if (dest_threshold == empty_String) {
        dest_threshold = messageLoggerDefaults->threshold(filename);
      }
      if (dest_threshold == empty_String)
        dest_threshold = COMMON_DEFAULT_THRESHOLD;
      ELseverityLevel threshold_sev(dest_threshold);
      dest_ctrl->setThreshold(threshold_sev);
      if (threshold_sev <= ELseverityLevel::ELsev_success) {
        edm::MessageDrop::debugAlwaysSuppressed = false;
      }
      if (threshold_sev <= ELseverityLevel::ELsev_info) {
        edm::MessageDrop::infoAlwaysSuppressed = false;
      }
      if (threshold_sev <= ELseverityLevel::ELsev_fwkInfo) {
        edm::MessageDrop::fwkInfoAlwaysSuppressed = false;
      }
      if (threshold_sev <= ELseverityLevel::ELsev_warning) {
        edm::MessageDrop::warningAlwaysSuppressed = false;
      }

      // establish this destination's limit/interval/timespan for each category:
      for (vString::const_iterator id_it = categories.begin(); id_it != categories.end(); ++id_it) {
        String msgID = *id_it;
        PSet default_category_pset = getAparameter<PSet>(default_pset, msgID, empty_PSet);
        PSet category_pset = getAparameter<PSet>(dest_pset, msgID, default_category_pset);

        int category_default_limit = getAparameter<int>(default_category_pset, "limit", NO_VALUE_SET);
        int limit = getAparameter<int>(category_pset, "limit", category_default_limit);
        if (limit == NO_VALUE_SET)
          limit = dest_default_limit;
        int category_default_interval = getAparameter<int>(default_category_pset, "reportEvery", NO_VALUE_SET);
        int interval = getAparameter<int>(category_pset, "reportEvery", category_default_interval);
        if (interval == NO_VALUE_SET)
          interval = dest_default_interval;
        int category_default_timespan = getAparameter<int>(default_category_pset, "timespan", NO_VALUE_SET);
        int timespan = getAparameter<int>(category_pset, "timespan", category_default_timespan);
        if (timespan == NO_VALUE_SET)
          timespan = dest_default_timespan;

        const std::string& category = msgID;
        if (limit == NO_VALUE_SET) {
          limit = messageLoggerDefaults->limit(filename, category);
        }
        if (interval == NO_VALUE_SET) {
          interval = messageLoggerDefaults->reportEvery(filename, category);
        }
        if (timespan == NO_VALUE_SET) {
          timespan = messageLoggerDefaults->timespan(filename, category);
        }

        if (limit != NO_VALUE_SET) {
          if (limit < 0)
            limit = 2000000000;
          dest_ctrl->setLimit(msgID, limit);
        }  // change log 2a, 2b
        if (interval != NO_VALUE_SET) {
          dest_ctrl->setInterval(msgID, interval);
        }  // change log 6
        if (timespan != NO_VALUE_SET) {
          if (timespan < 0)
            timespan = 2000000000;
          dest_ctrl->setTimespan(msgID, timespan);
        }

      }  // for

      // establish this destination's limit for each severity:
      for (vString::const_iterator sev_it = severities.begin(); sev_it != severities.end(); ++sev_it) {
        String sevID = *sev_it;
        ELseverityLevel severity(sevID);

        PSet default_sev_pset = getAparameter<PSet>(default_pset, sevID, empty_PSet);
        PSet sev_pset = getAparameter<PSet>(dest_pset, sevID, default_sev_pset);
        int limit = getAparameter<int>(sev_pset, "limit", NO_VALUE_SET);
        if (limit == NO_VALUE_SET) {
          limit = messageLoggerDefaults->sev_limit(filename, sevID);
        }
        if (limit != NO_VALUE_SET) {
          if (limit < 0)
            limit = 2000000000;
          dest_ctrl->setLimit(severity, limit);
        }
        int interval = getAparameter<int>(sev_pset, "reportEvery", NO_VALUE_SET);
        if (interval == NO_VALUE_SET) {
          interval = messageLoggerDefaults->sev_reportEvery(filename, sevID);
        }
        if (interval != NO_VALUE_SET)
          dest_ctrl->setInterval(severity, interval);
        // change log 2
        int timespan = getAparameter<int>(sev_pset, "timespan", NO_VALUE_SET);
        if (timespan == NO_VALUE_SET) {
          timespan = messageLoggerDefaults->sev_timespan(filename, sevID);
        }
        if (timespan != NO_VALUE_SET) {
          if (timespan < 0)
            timespan = 2000000000;
          dest_ctrl->setTimespan(severity, timespan);
        }
      }  // for

      // establish this destination's linebreak policy:
      bool noLineBreaks_default = getAparameter<bool>(default_pset, "noLineBreaks", false);
      bool noLineBreaks = getAparameter<bool>(dest_pset, "noLineBreaks", noLineBreaks_default);
      if (noLineBreaks) {
        dest_ctrl->setLineLength(32000);
      } else {
        int lenDef = 80;
        int lineLen_default = getAparameter<int>(default_pset, "lineLength", lenDef);
        int lineLen = getAparameter<int>(dest_pset, "lineLength", lineLen_default);
        if (lineLen != lenDef) {
          dest_ctrl->setLineLength(lineLen);
        }
      }

      // if indicated, suppress time stamps in this destination's output
      bool suppressTime_default = getAparameter<bool>(default_pset, "noTimeStamps", false);
      bool suppressTime = getAparameter<bool>(dest_pset, "noTimeStamps", suppressTime_default);
      if (suppressTime) {
        dest_ctrl->suppressTime();
      }

    }  // MessageLoggerScribe::configure_dest()

    void MessageLoggerScribe::configure_ordinary_destinations() {
      vString empty_vString;
      String empty_String;
      PSet empty_PSet;

      // Initialize unversal suppression variables
      MessageDrop::debugAlwaysSuppressed = true;
      MessageDrop::infoAlwaysSuppressed = true;
      MessageDrop::fwkInfoAlwaysSuppressed = true;
      MessageDrop::warningAlwaysSuppressed = true;

      // grab list of destinations:
      vString destinations = getAparameter<vString>(*job_pset_p, "destinations", empty_vString);

      // Use the default list of destinations if and only if the grabbed list is
      // empty
      if (destinations.empty()) {
        destinations = messageLoggerDefaults->destinations;
      }

      // dial down the early destination if other dest's are supplied:
      if (!destinations.empty())
        early_dest->setThreshold(ELhighestSeverity);

      // establish each destination:
      for (vString::const_iterator it = destinations.begin(); it != destinations.end(); ++it) {
        String filename = *it;
        String psetname = filename;

        // check that this destination is not just a placeholder
        PSet dest_pset = getAparameter<PSet>(*job_pset_p, psetname, empty_PSet);
        bool is_placeholder = getAparameter<bool>(dest_pset, "placeholder", false);
        if (is_placeholder)
          continue;

        // Modify the file name if extension or name is explicitly specified

        // Although for an ordinary destination there is no output attribute
        // for the cfg (you can use filename instead) we provide output() for
        // uniformity with the statistics destinations.  The "right way" to
        // work this would have been to provide a filename() method, along with
        // an extension() method.  We recognize the potential name confusion here
        // (filename(filename))!

        // Determine the destination file name to use if no explicit filename is
        // supplied in the cfg.
        String filename_default = getAparameter<String>(dest_pset, "output", empty_String);
        if (filename_default == empty_String) {
          filename_default = messageLoggerDefaults->output(psetname);
          if (filename_default == empty_String) {
            filename_default = filename;
          }
        }

        String explicit_filename = getAparameter<String>(dest_pset, "filename", filename_default);
        if (explicit_filename != empty_String)
          filename = explicit_filename;
        String explicit_extension = getAparameter<String>(dest_pset, "extension", empty_String);
        if (explicit_extension != empty_String) {
          if (explicit_extension[0] == '.') {
            filename += explicit_extension;
          } else {
            filename = filename + "." + explicit_extension;
          }
        }

        // Attach a default extension of .log if there is no extension on a file
        // change log 18 - this had been done in concert with attaching destination

        std::string actual_filename = filename;
        if ((filename != "cout") && (filename != "cerr")) {
          const std::string::size_type npos = std::string::npos;
          if (filename.find('.') == npos) {
            actual_filename += ".log";
          }
        }

        // Check that this is not a duplicate name
        if (stream_ps.find(actual_filename) != stream_ps.end()) {
          if (clean_slate_configuration) {
            LogError("duplicateDestination")
                << "Duplicate name for a MessageLogger Destination: " << actual_filename << "\n"
                << "Only the first configuration instructions are used";
            continue;
          } else {
            LogWarning("duplicateDestination")
                << "Duplicate name for a MessageLogger Destination: " << actual_filename << "\n"
                << "Only original configuration instructions are used";
            continue;
          }
        }

        ordinary_destination_filenames.push_back(actual_filename);

        // attach the current destination, keeping a control handle to it:
        std::shared_ptr<ELdestination> dest_ctrl;
        if (actual_filename == "cout") {
          dest_ctrl = std::make_shared<ELoutput>(std::cout);
          admin_p->attach(dest_ctrl);
          stream_ps["cout"] = &std::cout;
        } else if (actual_filename == "cerr") {
          early_dest->setThreshold(ELzeroSeverity);
          dest_ctrl = early_dest;
          stream_ps["cerr"] = &std::cerr;
        } else {
          auto os_sp = std::make_shared<std::ofstream>(actual_filename.c_str());
          file_ps.push_back(os_sp);
          dest_ctrl = std::make_shared<ELoutput>(*os_sp);
          admin_p->attach(dest_ctrl);
          stream_ps[actual_filename] = os_sp.get();
        }

        // now configure this destination:
        configure_dest(dest_ctrl, psetname);

      }  // for [it = destinations.begin() to end()]

    }  // configure_ordinary_destinations

    void MessageLoggerScribe::configure_statistics() {
      vString empty_vString;
      String empty_String;
      PSet empty_PSet;

      // grab list of statistics destinations:
      vString statistics = getAparameter<vString>(*job_pset_p, "statistics", empty_vString);

      bool no_statistics_configured = statistics.empty();

      if (no_statistics_configured) {
        // Read the list of staistics destinations from hardwired defaults,
        // but only if there is also no list of ordinary destinations.
        // (If a cfg specifies destinations, and no statistics, assume that
        // is what the user wants.)
        vString destinations = getAparameter<vString>(*job_pset_p, "destinations", empty_vString);
        if (destinations.empty()) {
          statistics = messageLoggerDefaults->statistics;
          no_statistics_configured = statistics.empty();
        }
      }

      // establish each statistics destination:
      for (vString::const_iterator it = statistics.begin(); it != statistics.end(); ++it) {
        String statname = *it;
        const String& psetname = statname;

        // check that this destination is not just a placeholder // change log 20
        PSet stat_pset = getAparameter<PSet>(*job_pset_p, psetname, empty_PSet);
        bool is_placeholder = getAparameter<bool>(stat_pset, "placeholder", false);
        if (is_placeholder)
          continue;

        // Determine the destination file name
        String filename = getAparameter<String>(stat_pset, "output", empty_String);
        if (filename == empty_String) {
          filename = messageLoggerDefaults->output(psetname);
          if (filename == empty_String) {
            filename = statname;
          }
        }

        // Modify the file name if extension or name is explicitly specified
        // change log 14 -- probably suspenders and a belt, because ouput option
        // is present, but uniformity is nice.

        String explicit_filename = getAparameter<String>(stat_pset, "filename", filename);
        if (explicit_filename != empty_String)
          filename = explicit_filename;
        String explicit_extension = getAparameter<String>(stat_pset, "extension", empty_String);
        if (explicit_extension != empty_String) {
          if (explicit_extension[0] == '.') {
            filename += explicit_extension;
          } else {
            filename = filename + "." + explicit_extension;
          }
        }

        // Attach a default extension of .log if there is no extension on a file
        // this had been done in concert with attaching destination

        std::string actual_filename = filename;
        if ((filename != "cout") && (filename != "cerr")) {
          const std::string::size_type npos = std::string::npos;
          if (filename.find('.') == npos) {
            actual_filename += ".log";
          }
        }

        // Check that this is not a duplicate name -
        // unless it is an ordinary destination (which stats can share)
        if (!search_all(ordinary_destination_filenames, actual_filename)) {
          if (stream_ps.find(actual_filename) != stream_ps.end()) {
            if (clean_slate_configuration) {
              throw edm::Exception(edm::errors::Configuration)
                  << "Duplicate name for a MessageLogger Statistics Destination: " << actual_filename << "\n";
            } else {
              LogWarning("duplicateDestination")
                  << "Duplicate name for a MessageLogger Statistics Destination: " << actual_filename << "\n"
                  << "Only original configuration instructions are used";
              continue;
            }
          }
        }

        // create (if statistics file does not match any destination file name)
        // or note (if statistics file matches a destination file name) the ostream.
        // But if no statistics destinations were provided in the config, do not
        // create a new destination for this hardwired statistics - only act if
        // it is matches a destination.
        bool statistics_destination_is_real = !no_statistics_configured;
        std::ostream* os_p;
        if (stream_ps.find(actual_filename) == stream_ps.end()) {
          if (actual_filename == "cout") {
            os_p = &std::cout;
          } else if (actual_filename == "cerr") {
            os_p = &std::cerr;
          } else {
            auto os_sp = std::make_shared<std::ofstream>(actual_filename.c_str());
            file_ps.push_back(os_sp);
            os_p = os_sp.get();
          }
          stream_ps[actual_filename] = os_p;
        } else {
          statistics_destination_is_real = true;
          os_p = stream_ps[actual_filename];
        }

        if (statistics_destination_is_real) {
          // attach the statistics destination, keeping a control handle to it:
          auto stat = std::make_shared<ELstatistics>(*os_p);
          admin_p->attach(stat);
          statisticsDestControls.push_back(stat);
          bool reset = getAparameter<bool>(stat_pset, "reset", false);
          statisticsResets.push_back(reset);

          // now configure this destination:
          configure_dest(stat, psetname);

          // and suppress the desire to do an extra termination summary just because
          // of end-of-job info messages
          stat->noTerminationSummary();
        }

      }  // for [it = statistics.begin() to end()]

    }  // configure_statistics

    void MessageLoggerScribe::parseCategories(std::string const& s, std::vector<std::string>& cats) {
      const std::string::size_type npos = std::string::npos;
      std::string::size_type i = 0;
      while (i != npos) {
        std::string::size_type j = s.find('|', i);
        cats.push_back(s.substr(i, j - i));
        i = j;
        while ((i != npos) && (s[i] == '|'))
          ++i;
        // the above handles cases of || and also | at end of string
      }
      // Note:  This algorithm assigns, as desired, one null category if it
      //        encounters an empty categories string
    }

    void MessageLoggerScribe::triggerStatisticsSummaries() {
      assert(statisticsDestControls.size() == statisticsResets.size());
      for (unsigned int i = 0; i != statisticsDestControls.size(); ++i) {
        statisticsDestControls[i]->summary(0);
        if (statisticsResets[i])
          statisticsDestControls[i]->wipe();
      }
    }

    void MessageLoggerScribe::triggerFJRmessageSummary(std::map<std::string, double>& sm) {
      if (statisticsDestControls.empty()) {
        sm["NoStatisticsDestinationsConfigured"] = 0.0;
      } else {
        statisticsDestControls[0]->summaryForJobReport(sm);
      }
    }

  }  // end of namespace service
}  // end of namespace edm
