// ----------------------------------------------------------------------
//
// ThreadSafeLogMessageLoggerScribe.cc
//
// NOTE: This was originally copied from MessageLoggerScribe but removed
//  the potential use of the ThreadQueue. MessageLoggerScribe was not
//  modified since it was decided we might have to keep the old behaviour
//  around for 'legacy' reasons.
//
// ----------------------------------------------------------------------

#include "FWCore/MessageService/src/ThreadSafeLogMessageLoggerScribe.h"
#include "FWCore/MessageService/src/ELadministrator.h"
#include "FWCore/MessageService/src/ELoutput.h"
#include "FWCore/MessageService/src/ELstatistics.h"

#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageService/src/ConfigurationHandshake.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"      // change log 37
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"  // change log 37

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterWildcardWithSpecifics.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <csignal>

using std::cerr;
using namespace edm::messagelogger;

namespace edm {
  namespace service {

    ThreadSafeLogMessageLoggerScribe::ThreadSafeLogMessageLoggerScribe()
        : m_admin_p(std::make_shared<ELadministrator>()),
          m_early_dest(m_admin_p->attach(std::make_shared<ELoutput>(std::cerr, false))),
          m_clean_slate_configuration(true),
          m_active(true),
          m_purge_mode(false),
          m_count(0),
          m_messageBeingSent(false),
          m_waitingThreshold(100),
          m_tooManyWaitingMessagesCount(0) {}

    ThreadSafeLogMessageLoggerScribe::~ThreadSafeLogMessageLoggerScribe() {
      //if there are any waiting message, finish them off
      ErrorObj* errorobj_p = nullptr;
      std::vector<std::string> categories;
      while (m_waitingMessages.try_pop(errorobj_p)) {
        if (not m_purge_mode) {
          categories.clear();
          parseCategories(errorobj_p->xid().id, categories);
          for (unsigned int icat = 0; icat < categories.size(); ++icat) {
            errorobj_p->setID(categories[icat]);
            m_admin_p->log(*errorobj_p);  // route the message text
          }
        }
        delete errorobj_p;
      }

      m_admin_p->finish();
    }

    void ThreadSafeLogMessageLoggerScribe::runCommand(  // changeLog 32
        MessageLoggerQ::OpCode opcode,
        void* operand) {
      switch (opcode) {  // interpret the work item
        default: {
          assert(false);  // can't happen (we certainly hope!)
          break;
        }
        case MessageLoggerQ::END_THREAD: {
          break;
        }
        case MessageLoggerQ::LOG_A_MESSAGE: {
          ErrorObj* errorobj_p = static_cast<ErrorObj*>(operand);
          try {
            if (m_active && !m_purge_mode) {
              log(errorobj_p);
            }
          } catch (cms::Exception& e) {
            ++m_count;
            std::cerr << "ThreadSafeLogMessageLoggerScribe caught " << m_count << " cms::Exceptions, text = \n"
                      << e.what() << "\n";

            if (m_count > 25) {
              cerr << "MessageLogger will no longer be processing "
                   << "messages due to errors (entering purge mode).\n";
              m_purge_mode = true;
            }
          } catch (...) {
            std::cerr << "ThreadSafeLogMessageLoggerScribe caught an unknown exception and "
                      << "will no longer be processing "
                      << "messages. (entering purge mode)\n";
            m_purge_mode = true;
          }
          break;
        }
        case MessageLoggerQ::CONFIGURE: {  // changelog 17
          auto job_pset_p = std::unique_ptr<edm::ParameterSet>(
              static_cast<edm::ParameterSet*>(operand));  // propagate_const<T> has no reset() function
          validate(*job_pset_p);
          configure_errorlog(*job_pset_p);
          break;
        }
        case MessageLoggerQ::SUMMARIZE: {
          assert(operand == nullptr);
          try {
            triggerStatisticsSummaries();
          } catch (cms::Exception& e) {
            std::cerr << "ThreadSafeLogMessageLoggerScribe caught exception "
                      << "during summarize:\n"
                      << e.what() << "\n";
          } catch (...) {
            std::cerr << "ThreadSafeLogMessageLoggerScribe caught unkonwn exception type "
                      << "during summarize. (Ignored)\n";
          }
          break;
        }
        case MessageLoggerQ::JOBMODE: {  // change log 24
          std::string* jobMode_p = static_cast<std::string*>(operand);
          JobMode jm = MessageLoggerDefaults::mode(*jobMode_p);
          m_messageLoggerDefaults = value_ptr<MessageLoggerDefaults>(new MessageLoggerDefaults(jm));
          // Note - since m_messageLoggerDefaults is a value_ptr,
          //        there is no concern about deleting here.
          delete jobMode_p;  // dispose of the message text
                             // which will have been new-ed
                             // in MessageLogger.cc (service version)
          break;
        }
        case MessageLoggerQ::SHUT_UP: {
          assert(operand == nullptr);
          m_active = false;
          break;
        }
        case MessageLoggerQ::FLUSH_LOG_Q: {  // changelog 26
          break;
        }
        case MessageLoggerQ::GROUP_STATS: {  // change log 27
          std::string* cat_p = static_cast<std::string*>(operand);
          ELstatistics::noteGroupedCategory(*cat_p);
          delete cat_p;  // dispose of the message text
          break;
        }
        case MessageLoggerQ::FJR_SUMMARY: {  // changelog 29
          std::map<std::string, double>* smp = static_cast<std::map<std::string, double>*>(operand);
          triggerFJRmessageSummary(*smp);
          break;
        }
      }  // switch

    }  // ThreadSafeLogMessageLoggerScribe::runCommand(opcode, operand)

    void ThreadSafeLogMessageLoggerScribe::log(ErrorObj* errorobj_p) {
      bool expected = false;
      std::unique_ptr<ErrorObj> obj(errorobj_p);
      if (m_messageBeingSent.compare_exchange_strong(expected, true)) {
        std::vector<std::string> categories;
        parseCategories(errorobj_p->xid().id, categories);
        for (unsigned int icat = 0; icat < categories.size(); ++icat) {
          errorobj_p->setID(categories[icat]);
          m_admin_p->log(*errorobj_p);  // route the message text
        }
        //process any waiting messages
        errorobj_p = nullptr;
        while (not m_purge_mode and m_waitingMessages.try_pop(errorobj_p)) {
          obj.reset(errorobj_p);
          categories.clear();
          parseCategories(errorobj_p->xid().id, categories);
          for (unsigned int icat = 0; icat < categories.size(); ++icat) {
            errorobj_p->setID(categories[icat]);
            m_admin_p->log(*errorobj_p);  // route the message text
          }
        }
        m_messageBeingSent.store(false);
      } else {
        if (m_waitingMessages.unsafe_size() < m_waitingThreshold) {
          obj.release();
          m_waitingMessages.push(errorobj_p);
        } else {
          ++m_tooManyWaitingMessagesCount;
        }
      }
    }

    namespace {
      bool usingOldConfig(edm::ParameterSet const& pset) {
        if (not pset.exists("files") and
            ((pset.exists("destinations") or pset.existsAs<std::vector<std::string>>("statistics", true) or
              pset.existsAs<std::vector<std::string>>("statistics", false) or pset.exists("categories")))) {
          return true;
        }
        return false;
      }

      std::set<std::string> findCategoriesInDestination(edm::ParameterSet const& pset) {
        auto psets = pset.getParameterNamesForType<edm::ParameterSet>(false);
        auto itFound = std::find(psets.begin(), psets.end(), "default");
        if (itFound != psets.end()) {
          psets.erase(itFound);
        }

        return std::set<std::string>(psets.begin(), psets.end());
      }
      std::vector<std::string> findAllCategories(edm::ParameterSet const& pset) {
        std::set<std::string> categories;

        auto psets = pset.getParameterNamesForType<edm::ParameterSet>(false);
        auto itFound = std::find(psets.begin(), psets.end(), "default");
        if (itFound != psets.end()) {
          categories = findCategoriesInDestination(pset.getUntrackedParameter<edm::ParameterSet>("default"));
          psets.erase(itFound);
        }

        itFound = std::find(psets.begin(), psets.end(), "cout");
        if (itFound != psets.end()) {
          categories.merge(findCategoriesInDestination(pset.getUntrackedParameter<edm::ParameterSet>("cout")));
          psets.erase(itFound);
        }

        itFound = std::find(psets.begin(), psets.end(), "cerr");
        if (itFound != psets.end()) {
          categories.merge(findCategoriesInDestination(pset.getUntrackedParameter<edm::ParameterSet>("cerr")));
          psets.erase(itFound);
        }

        auto const& files = pset.getUntrackedParameter<edm::ParameterSet>("files");
        for (auto const& name : files.getParameterNamesForType<edm::ParameterSet>(false)) {
          categories.merge(findCategoriesInDestination(files.getUntrackedParameter<edm::ParameterSet>(name)));
        }
        categories.insert(psets.begin(), psets.end());

        return std::vector<std::string>(categories.begin(), categories.end());
      }

    }  // namespace

    std::string ThreadSafeLogMessageLoggerScribe::destinationFileName(edm::ParameterSet const& dest_pset,
                                                                      std::string const& psetname) const {
      // Determine the destination file name to use if no explicit filename is
      // supplied in the cfg.
      std::string const empty_String;
      std::string filename = psetname;
      std::string filename_default = getAparameter<std::string>(dest_pset, "output", empty_String);
      if (filename_default == empty_String) {
        filename_default = m_messageLoggerDefaults->output(psetname);  // change log 31
        if (filename_default == empty_String) {
          filename_default = filename;
        }
      }

      std::string explicit_filename = getAparameter<std::string>(dest_pset, "filename", filename_default);
      if (explicit_filename != empty_String)
        filename = explicit_filename;
      std::string explicit_extension = getAparameter<std::string>(dest_pset, "extension", empty_String);
      if (explicit_extension != empty_String) {
        if (explicit_extension[0] == '.') {
          filename += explicit_extension;
        } else {
          filename = filename + "." + explicit_extension;
        }
      }

      // Attach a default extension of .log if there is no extension on a file
      if ((filename != "cout") && (filename != "cerr")) {
        if (filename.find('.') == std::string::npos) {
          filename += ".log";
        }
      }
      return filename;
    }

    void ThreadSafeLogMessageLoggerScribe::configure_errorlog_new(edm::ParameterSet& job_pset) {
      {
        auto preconfiguration_message =
            job_pset.getUntrackedParameter<std::string>("generate_preconfiguration_message");
        if (not preconfiguration_message.empty()) {
          // To test a preconfiguration message without first going thru the
          // configuration we are about to do, we issue the message (so it sits
          // on the queue), then copy the processing that the LOG_A_MESSAGE case
          // does.  We suppress the timestamp to allow for automated unit testing.
          m_early_dest->suppressTime();
          LogError("preconfiguration") << preconfiguration_message;
        }
      }
      if (!m_stream_ps.empty()) {
        LogWarning("multiLogConfig") << "The message logger has been configured multiple times";
        m_clean_slate_configuration = false;  // Change Log 22
      }
      m_waitingThreshold = job_pset.getUntrackedParameter<unsigned int>("waiting_threshold");

      auto defaults = parseDefaults(job_pset);
      auto categories = findAllCategories(job_pset);

      // Initialize unversal suppression variables
      MessageDrop::debugAlwaysSuppressed = true;
      MessageDrop::infoAlwaysSuppressed = true;
      MessageDrop::fwkInfoAlwaysSuppressed = true;
      MessageDrop::warningAlwaysSuppressed = true;

      m_early_dest->setThreshold(ELhighestSeverity);

      auto cout_dest = job_pset.getUntrackedParameter<edm::ParameterSet>("cout");
      if (cout_dest.getUntrackedParameter<bool>("enable")) {
        auto dest_ctrl = makeDestinationCtrl("cout");
        configure_dest(job_pset, defaults, categories, dest_ctrl, cout_dest, "cout");
      }

      auto cerr_dest = job_pset.getUntrackedParameter<edm::ParameterSet>("cerr");
      if (cerr_dest.getUntrackedParameter<bool>("enable")) {
        auto dest_ctrl = makeDestinationCtrl("cerr");
        configure_dest(job_pset, defaults, categories, dest_ctrl, cerr_dest, "cerr");
      }

      auto const& files = job_pset.getUntrackedParameter<edm::ParameterSet>("files");
      for (auto const& name : files.getParameterNamesForType<edm::ParameterSet>(false)) {
        auto const& dest_pset = files.getUntrackedParameter<edm::ParameterSet>(name);
        auto const actual_filename = destinationFileName(dest_pset, name);

        // Check that this is not a duplicate name
        if (m_stream_ps.find(actual_filename) != m_stream_ps.end()) {
          if (m_clean_slate_configuration) {
            throw cms::Exception("DuplicateDestination")
                << "Duplicate name for a MessageLogger Destination: " << actual_filename << "\n"
                << "Please modify the configuration to use unique file names.";
          } else {
            LogWarning("duplicateDestination")
                << "Duplicate name for a MessageLogger Destination: " << actual_filename << "\n"
                << "Only original configuration instructions are used";
            continue;
          }
        }

        auto dest_ctrl = makeDestinationCtrl(actual_filename);
        configure_dest(job_pset, defaults, categories, dest_ctrl, dest_pset, name);
      }
      //NOTE: statistics destinations MUST BE last in the list else they can be fooled into
      // thinking a message has been ignored just because a later destination which uses it
      // falls later in the list.
      for (auto const& name : files.getParameterNamesForType<edm::ParameterSet>(false)) {
        auto const& dest_pset = files.getUntrackedParameter<edm::ParameterSet>(name);
        auto const actual_filename = destinationFileName(dest_pset, name);
        if (getAparameter<bool>(dest_pset, "enableStatistics", false)) {
          configure_statistics_dest(job_pset, defaults, categories, dest_pset, name, actual_filename);
        }
      }
      if (cout_dest.getUntrackedParameter<bool>("enable") and
          getAparameter<bool>(cout_dest, "enableStatistics", true)) {
        configure_statistics_dest(job_pset, defaults, categories, cout_dest, "cout", "cout");
      }
      if (cerr_dest.getUntrackedParameter<bool>("enable") and
          getAparameter<bool>(cerr_dest, "enableStatistics", true)) {
        configure_statistics_dest(job_pset, defaults, categories, cerr_dest, "cerr", "cerr");
      }
    }

    void ThreadSafeLogMessageLoggerScribe::configure_errorlog(edm::ParameterSet& job_pset) {
      if (not usingOldConfig(job_pset)) {
        configure_errorlog_new(job_pset);
        return;
      }
      const vString empty_vString;
      const std::string empty_String;
      const edm::ParameterSet empty_PSet;

      // The following is present to test pre-configuration message handling:
      std::string preconfiguration_message =
          getAparameter<std::string>(job_pset, "generate_preconfiguration_message", empty_String);
      if (preconfiguration_message != empty_String) {
        // To test a preconfiguration message without first going thru the
        // configuration we are about to do, we issue the message (so it sits
        // on the queue), then copy the processing that the LOG_A_MESSAGE case
        // does.  We suppress the timestamp to allow for automated unit testing.
        m_early_dest->suppressTime();
        LogError("preconfiguration") << preconfiguration_message;
      }

      if (!m_stream_ps.empty()) {
        LogWarning("multiLogConfig") << "The message logger has been configured multiple times";
        m_clean_slate_configuration = false;  // Change Log 22
      }
      m_waitingThreshold = getAparameter<unsigned int>(job_pset, "waiting_threshold", 100);
      auto defaults = parseDefaults(job_pset);
      // grab list of categories
      vString categories = getAparameter<vString>(job_pset, "categories", empty_vString);
      // grab list of hardwired categories (hardcats) -- these are to be added
      // to the list of categories -- change log 24
      {
        std::vector<std::string> hardcats = m_messageLoggerDefaults->categories;
        // combine the lists, not caring about possible duplicates (for now)
        copy_all(hardcats, std::back_inserter(categories));
      }  // no longer need hardcats

      auto destination_names = configure_ordinary_destinations(job_pset, defaults, categories);
      configure_statistics(job_pset, defaults, categories, destination_names);
    }  // ThreadSafeLogMessageLoggerScribe::configure_errorlog()

    std::shared_ptr<ELdestination> ThreadSafeLogMessageLoggerScribe::makeDestinationCtrl(std::string const& filename) {
      std::shared_ptr<ELdestination> dest_ctrl;
      if (filename == "cout") {
        dest_ctrl = m_admin_p->attach(std::make_shared<ELoutput>(std::cout));
        m_stream_ps["cout"] = &std::cout;
      } else if (filename == "cerr") {
        m_early_dest->setThreshold(ELzeroSeverity);
        dest_ctrl = m_early_dest;
        m_stream_ps["cerr"] = &std::cerr;
      } else {
        auto os_sp = std::make_shared<std::ofstream>(filename.c_str());
        m_file_ps.push_back(os_sp);
        dest_ctrl = m_admin_p->attach(std::make_shared<ELoutput>(*os_sp));
        m_stream_ps[filename] = os_sp.get();
      }
      return dest_ctrl;
    }

    namespace {
      void setGlobalThresholds(ELseverityLevel threshold_sev) {
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
      }
    }  // namespace

    ThreadSafeLogMessageLoggerScribe::ConfigurableDefaults ThreadSafeLogMessageLoggerScribe::parseDefaults(
        edm::ParameterSet const& job_pset) {
      const edm::ParameterSet empty_PSet;
      ThreadSafeLogMessageLoggerScribe::ConfigurableDefaults returnValue;
      // grab default limit/interval/timespan common to all destinations/categories:
      edm::ParameterSet default_pset = getAparameter<edm::ParameterSet>(job_pset, "default", empty_PSet);
      returnValue.limit_ = getAparameter<int>(
          default_pset, "limit", ThreadSafeLogMessageLoggerScribe::ConfigurableDefaults::COMMON_DEFAULT_LIMIT);
      returnValue.reportEvery_ = getAparameter<int>(
          default_pset, "reportEvery", ThreadSafeLogMessageLoggerScribe::ConfigurableDefaults::COMMON_DEFAULT_INTERVAL);
      returnValue.timespan_ = getAparameter<int>(
          default_pset, "timespan", ThreadSafeLogMessageLoggerScribe::ConfigurableDefaults::COMMON_DEFAULT_TIMESPAN);
      std::string default_threshold = getAparameter<std::string>(job_pset, "threshold", std::string());
      returnValue.threshold_ = getAparameter<std::string>(default_pset, "threshold", default_threshold);
      returnValue.noLineBreaks_ = getAparameter<bool>(default_pset, "noLineBreaks", false);
      returnValue.lineLength_ = getAparameter<int>(default_pset, "lineLength", 80);
      returnValue.noTimeStamps_ = getAparameter<bool>(default_pset, "noTimeStamps", false);

      return returnValue;
    }

    void ThreadSafeLogMessageLoggerScribe::configure_dest(edm::ParameterSet const& job_pset,
                                                          ConfigurableDefaults const& defaults,
                                                          vString const& categories,
                                                          std::shared_ptr<ELdestination> dest_ctrl,
                                                          edm::ParameterSet const& dest_pset,
                                                          std::string const& filename) {
      vString const empty_vString;
      edm::ParameterSet const empty_PSet;
      std::string const empty_String;

      // Defaults:							// change log 3a
      const std::string COMMON_DEFAULT_THRESHOLD = "INFO";

      vString const severities = {{"WARNING", "INFO", "FWKINFO", "ERROR", "DEBUG"}};

      // grab default threshold common to all destinations
      std::string const default_threshold = getAparameter<std::string>(job_pset, "threshold", empty_String);
      // change log 3a
      // change log 24

      // grab default limit/interval/timespan common to all destinations/categories:
      edm::ParameterSet const default_pset = getAparameter<edm::ParameterSet>(job_pset, "default", empty_PSet);

      // See if this is just a placeholder			// change log 9
      bool is_placeholder = getAparameter<bool>(dest_pset, "placeholder", false);
      if (is_placeholder)
        return;

      // grab this destination's default limit/interval/timespan:
      edm::ParameterSet dest_default_pset = getAparameter<edm::ParameterSet>(dest_pset, "default", empty_PSet);
      int dest_default_limit = getAparameter<int>(dest_default_pset, "limit", defaults.limit_);
      int dest_default_interval = getAparameter<int>(dest_default_pset, "reportEvery", defaults.reportEvery_);
      // change log 6
      int dest_default_timespan = getAparameter<int>(dest_default_pset, "timespan", defaults.timespan_);
      // change log 1a
      if (dest_default_limit != defaults.NO_VALUE_SET) {
        if (dest_default_limit < 0)
          dest_default_limit = 2000000000;
        dest_ctrl->setLimit("*", dest_default_limit);
      }                                                      // change log 1b, 2a, 2b
      if (dest_default_interval != defaults.NO_VALUE_SET) {  // change log 6
        dest_ctrl->setInterval("*", dest_default_interval);
      }
      if (dest_default_timespan != defaults.NO_VALUE_SET) {
        if (dest_default_timespan < 0)
          dest_default_timespan = 2000000000;
        dest_ctrl->setTimespan("*", dest_default_timespan);
      }  // change log 1b, 2a, 2b

      // establish this destination's threshold:
      std::string dest_threshold = getAparameter<std::string>(dest_pset, "threshold", default_threshold);
      if (dest_threshold == empty_String) {
        dest_threshold = default_threshold;
      }
      if (dest_threshold == empty_String) {  // change log 34
        dest_threshold = defaults.threshold_;
      }
      if (dest_threshold == empty_String) {
        dest_threshold = m_messageLoggerDefaults->threshold(filename);
      }
      if (dest_threshold == empty_String)
        dest_threshold = COMMON_DEFAULT_THRESHOLD;
      ELseverityLevel threshold_sev(dest_threshold);
      dest_ctrl->setThreshold(threshold_sev);
      // change log 37
      setGlobalThresholds(threshold_sev);

      // establish this destination's limit/interval/timespan for each category:
      for (vString::const_iterator id_it = categories.begin(); id_it != categories.end(); ++id_it) {
        const std::string& msgID = *id_it;
        edm::ParameterSet default_category_pset =
            getAparameter<edm::ParameterSet>(default_pset, msgID, empty_PSet);  // change log 5
        edm::ParameterSet category_pset = getAparameter<edm::ParameterSet>(dest_pset, msgID, default_category_pset);

        int category_default_limit = getAparameter<int>(default_category_pset, "limit", defaults.NO_VALUE_SET);
        int limit = getAparameter<int>(category_pset, "limit", category_default_limit);
        if (limit == defaults.NO_VALUE_SET)
          limit = dest_default_limit;
        // change log 7
        int category_default_interval = getAparameter<int>(default_category_pset, "reportEvery", defaults.NO_VALUE_SET);
        int interval = getAparameter<int>(category_pset, "reportEvery", category_default_interval);
        if (interval == defaults.NO_VALUE_SET)
          interval = dest_default_interval;
        // change log 6  and then 7
        int category_default_timespan = getAparameter<int>(default_category_pset, "timespan", defaults.NO_VALUE_SET);
        int timespan = getAparameter<int>(category_pset, "timespan", category_default_timespan);
        if (timespan == defaults.NO_VALUE_SET)
          timespan = dest_default_timespan;
        // change log 7

        const std::string& category = msgID;
        if (limit == defaults.NO_VALUE_SET) {  // change log 24
          limit = m_messageLoggerDefaults->limit(filename, category);
        }
        if (interval == defaults.NO_VALUE_SET) {  // change log 24
          interval = m_messageLoggerDefaults->reportEvery(filename, category);
        }
        if (timespan == defaults.NO_VALUE_SET) {  // change log 24
          timespan = m_messageLoggerDefaults->timespan(filename, category);
        }

        if (limit != defaults.NO_VALUE_SET) {
          if (limit < 0)
            limit = 2000000000;
          dest_ctrl->setLimit(msgID, limit);
        }  // change log 2a, 2b
        if (interval != defaults.NO_VALUE_SET) {
          dest_ctrl->setInterval(msgID, interval);
        }  // change log 6
        if (timespan != defaults.NO_VALUE_SET) {
          if (timespan < 0)
            timespan = 2000000000;
          dest_ctrl->setTimespan(msgID, timespan);
        }  // change log 2a, 2b

      }  // for

      // establish this destination's limit for each severity:
      for (vString::const_iterator sev_it = severities.begin(); sev_it != severities.end(); ++sev_it) {
        const std::string& sevID = *sev_it;
        ELseverityLevel severity(sevID);
        edm::ParameterSet default_sev_pset = getAparameter<edm::ParameterSet>(default_pset, sevID, empty_PSet);
        edm::ParameterSet sev_pset = getAparameter<edm::ParameterSet>(dest_pset, sevID, default_sev_pset);
        // change log 5
        int limit = getAparameter<int>(sev_pset, "limit", defaults.NO_VALUE_SET);
        if (limit == defaults.NO_VALUE_SET) {  // change log 24
          limit = m_messageLoggerDefaults->sev_limit(filename, sevID);
        }
        if (limit != defaults.NO_VALUE_SET) {
          if (limit < 0)
            limit = 2000000000;  // change log 38
          dest_ctrl->setLimit(severity, limit);
        }
        int interval = getAparameter<int>(sev_pset, "reportEvery", defaults.NO_VALUE_SET);
        if (interval == defaults.NO_VALUE_SET) {  // change log 24
          interval = m_messageLoggerDefaults->sev_reportEvery(filename, sevID);
        }
        if (interval != defaults.NO_VALUE_SET)
          dest_ctrl->setInterval(severity, interval);
        // change log 2
        int timespan = getAparameter<int>(sev_pset, "timespan", defaults.NO_VALUE_SET);
        if (timespan == defaults.NO_VALUE_SET) {  // change log 24
          timespan = m_messageLoggerDefaults->sev_timespan(filename, sevID);
        }
        if (timespan != defaults.NO_VALUE_SET) {
          if (timespan < 0)
            timespan = 2000000000;  // change log 38
          dest_ctrl->setTimespan(severity, timespan);
        }
      }  // for

      // establish this destination's linebreak policy:
      // change log 5
      bool noLineBreaks = getAparameter<bool>(dest_pset, "noLineBreaks", defaults.noLineBreaks_);
      if (noLineBreaks) {
        dest_ctrl->setLineLength(32000);
      } else {
        // change log 5
        int lineLen = getAparameter<int>(dest_pset, "lineLength", defaults.lineLength_);
        dest_ctrl->setLineLength(lineLen);
      }

      // if indicated, suppress time stamps in this destination's output
      bool suppressTime = getAparameter<bool>(dest_pset, "noTimeStamps", defaults.noTimeStamps_);
      if (suppressTime) {
        dest_ctrl->suppressTime();
      }

    }  // ThreadSafeLogMessageLoggerScribe::configure_dest()

    std::vector<std::string> ThreadSafeLogMessageLoggerScribe::configure_ordinary_destinations(
        edm::ParameterSet const& job_pset, ConfigurableDefaults const& defaults, vString const& categories) {
      vString const empty_vString;
      std::string const empty_String;
      edm::ParameterSet const empty_PSet;

      // Initialize unversal suppression variables
      MessageDrop::debugAlwaysSuppressed = true;
      MessageDrop::infoAlwaysSuppressed = true;
      MessageDrop::fwkInfoAlwaysSuppressed = true;
      MessageDrop::warningAlwaysSuppressed = true;

      // grab list of destinations:
      vString destinations = getAparameter<vString>(job_pset, "destinations", empty_vString);

      // Use the default list of destinations if and only if the grabbed list is
      // empty						 	// change log 24
      if (destinations.empty()) {
        destinations = m_messageLoggerDefaults->destinations;
      }

      // dial down the early destination if other dest's are supplied:
      if (!destinations.empty())
        m_early_dest->setThreshold(ELhighestSeverity);

      // establish each destination:
      std::vector<std::string> ordinary_destination_filenames;
      for (vString::const_iterator it = destinations.begin(); it != destinations.end(); ++it) {
        const std::string& filename = *it;
        const std::string& psetname = filename;

        // check that this destination is not just a placeholder // change log 11
        edm::ParameterSet dest_pset = getAparameter<edm::ParameterSet>(job_pset, psetname, empty_PSet);
        bool is_placeholder = getAparameter<bool>(dest_pset, "placeholder", false);
        if (is_placeholder)
          continue;

        // Modify the file name if extension or name is explicitly specified
        // change log 14

        // Although for an ordinary destination there is no output attribute
        // for the cfg (you can use filename instead) we provide output() for
        // uniformity with the statistics destinations.  The "right way" to
        // work this would have been to provide a filename() method, along with
        // an extension() method.  We recognize the potential name confusion here
        // (filename(filename))!

        auto const actual_filename = destinationFileName(dest_pset, psetname);

        // Check that this is not a duplicate name			// change log 18
        if (m_stream_ps.find(actual_filename) != m_stream_ps.end()) {
          if (m_clean_slate_configuration) {  // change log 22
                                              //        throw edm::Exception ( edm::errors::Configuration )
            LogError("duplicateDestination")  // change log 35
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
        std::shared_ptr<ELdestination> dest_ctrl = makeDestinationCtrl(actual_filename);
        // now configure this destination:
        configure_dest(job_pset, defaults, categories, dest_ctrl, dest_pset, psetname);

      }  // for [it = destinations.begin() to end()]

      return ordinary_destination_filenames;
    }  // configure_ordinary_destinations

    void ThreadSafeLogMessageLoggerScribe::configure_statistics_dest(edm::ParameterSet const& job_pset,
                                                                     ConfigurableDefaults const& defaults,
                                                                     vString const& categories,
                                                                     edm::ParameterSet const& stat_pset,
                                                                     std::string const& psetname,
                                                                     std::string const& filename) {
      auto os_p = m_stream_ps[filename];

      auto stat = std::make_shared<ELstatistics>(*os_p);
      m_admin_p->attach(stat);
      m_statisticsDestControls.push_back(stat);
      bool reset = getAparameter<bool>(stat_pset, "resetStatistics", false);
      if (not reset) {
        //check for old syntax
        reset = getAparameter<bool>(stat_pset, "reset", false);
      }
      m_statisticsResets.push_back(reset);

      // now configure this destination:
      configure_dest(job_pset, defaults, categories, stat, stat_pset, psetname);

      std::string dest_threshold = getAparameter<std::string>(stat_pset, "statisticsThreshold", std::string());
      if (not dest_threshold.empty()) {
        ELseverityLevel threshold_sev(dest_threshold);
        stat->setThreshold(threshold_sev);

        setGlobalThresholds(threshold_sev);
      }

      // and suppress the desire to do an extra termination summary just because
      // of end-of-job info messages
      stat->noTerminationSummary();
    }

    void ThreadSafeLogMessageLoggerScribe::configure_statistics(edm::ParameterSet const& job_pset,
                                                                ConfigurableDefaults const& defaults,
                                                                vString const& categories,
                                                                vString const& ordinary_destination_filenames) {
      vString const empty_vString;
      std::string const empty_String;
      edm::ParameterSet const empty_PSet;

      // grab list of statistics destinations:
      vString statistics = getAparameter<vString>(job_pset, "statistics", empty_vString);

      bool no_statistics_configured = statistics.empty();  // change log 24

      if (no_statistics_configured) {
        // Read the list of staistics destinations from hardwired defaults,
        // but only if there is also no list of ordinary destinations.
        // (If a cfg specifies destinations, and no statistics, assume that
        // is what the user wants.)
        vString destinations = getAparameter<vString>(job_pset, "destinations", empty_vString);
        if (destinations.empty()) {
          statistics = m_messageLoggerDefaults->statistics;
          no_statistics_configured = statistics.empty();
        } else {
          for (auto const& dest : destinations) {
            edm::ParameterSet stat_pset = getAparameter<edm::ParameterSet>(job_pset, dest, empty_PSet);
            if (getAparameter<bool>(stat_pset, "enableStatistics", false)) {
              statistics.push_back(dest);
            }
          }
        }
      }

      // establish each statistics destination:
      for (auto const& psetname : statistics) {
        // check that this destination is not just a placeholder // change log 20
        edm::ParameterSet stat_pset = getAparameter<edm::ParameterSet>(job_pset, psetname, empty_PSet);
        bool is_placeholder = getAparameter<bool>(stat_pset, "placeholder", false);
        if (is_placeholder)
          continue;

        // Determine the destination file name
        std::string filename = getAparameter<std::string>(stat_pset, "output", empty_String);
        if (filename == empty_String) {
          filename = m_messageLoggerDefaults->output(psetname);  // change log 31
          if (filename == empty_String) {
            filename = psetname;
          }
        }

        // Modify the file name if extension or name is explicitly specified
        // change log 14 -- probably suspenders and a belt, because ouput option
        // is present, but uniformity is nice.

        std::string explicit_filename = getAparameter<std::string>(stat_pset, "filename", filename);
        if (explicit_filename != empty_String)
          filename = explicit_filename;
        std::string explicit_extension = getAparameter<std::string>(stat_pset, "extension", empty_String);
        if (explicit_extension != empty_String) {
          if (explicit_extension[0] == '.') {
            filename += explicit_extension;
          } else {
            filename = filename + "." + explicit_extension;
          }
        }

        // Attach a default extension of .log if there is no extension on a file
        // change log 18 - this had been done in concert with attaching destination

        std::string actual_filename = filename;              // change log 4
        if ((filename != "cout") && (filename != "cerr")) {  // change log 23
          const std::string::size_type npos = std::string::npos;
          if (filename.find('.') == npos) {
            actual_filename += ".log";
          }
        }

        // Check that this is not a duplicate name -
        // unless it is an ordinary destination (which stats can share)
        if (!search_all(ordinary_destination_filenames, actual_filename)) {
          if (m_stream_ps.find(actual_filename) != m_stream_ps.end()) {
            if (m_clean_slate_configuration) {  // change log 22
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
        // it is matches a destination.  (shange log 24)
        bool statistics_destination_is_real = !no_statistics_configured;
        std::ostream* os_p;
        if (m_stream_ps.find(actual_filename) == m_stream_ps.end()) {
          if (actual_filename == "cout") {
            os_p = &std::cout;
          } else if (actual_filename == "cerr") {
            os_p = &std::cerr;
          } else {
            auto os_sp = std::make_shared<std::ofstream>(actual_filename.c_str());
            m_file_ps.push_back(os_sp);
            os_p = os_sp.get();
          }
          m_stream_ps[actual_filename] = os_p;
        } else {
          statistics_destination_is_real = true;  // change log 24
        }

        if (statistics_destination_is_real) {  // change log 24
                                               // attach the statistics destination, keeping a control handle to it:

          configure_statistics_dest(job_pset, defaults, categories, stat_pset, psetname, actual_filename);
        }

      }  // for [it = statistics.begin() to end()]

    }  // configure_statistics

    void ThreadSafeLogMessageLoggerScribe::parseCategories(std::string const& s, std::vector<std::string>& cats) {
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

    void ThreadSafeLogMessageLoggerScribe::triggerStatisticsSummaries() {
      assert(m_statisticsDestControls.size() == m_statisticsResets.size());
      for (unsigned int i = 0; i != m_statisticsDestControls.size(); ++i) {
        m_statisticsDestControls[i]->summary(m_tooManyWaitingMessagesCount.load());
        if (m_statisticsResets[i])
          m_statisticsDestControls[i]->wipe();
      }
      auto dropped = m_tooManyWaitingMessagesCount.load();
      if (m_statisticsDestControls.empty() and dropped != 0) {
        if (m_stream_ps.find("cerr") != m_stream_ps.end()) {
          std::cerr << "MessageLogger: dropped waiting message count " << dropped << "\n";
        }
        if (m_stream_ps.find("cout") != m_stream_ps.end()) {
          std::cout << "MessageLogger: dropped waiting message count " << dropped << "\n";
        }
      }
    }

    void ThreadSafeLogMessageLoggerScribe::triggerFJRmessageSummary(std::map<std::string, double>& sm)  // ChangeLog 29
    {
      if (m_statisticsDestControls.empty()) {
        sm["NoStatisticsDestinationsConfigured"] = 0.0;
      } else {
        m_statisticsDestControls[0]->summaryForJobReport(sm);
      }
    }

    namespace {
      void fillDescriptions(edm::ConfigurationDescriptions& config) {
        edm::ParameterSetDescription topDesc;

        topDesc.addUntracked<bool>("messageSummaryToJobReport", false);
        topDesc.addUntracked<std::string>("generate_preconfiguration_message", "");
        topDesc.addUntracked<unsigned int>("waiting_threshold", 100);
        topDesc.addUntracked<std::vector<std::string>>("suppressDebug", {});
        topDesc.addUntracked<std::vector<std::string>>("suppressInfo", {});
        topDesc.addUntracked<std::vector<std::string>>("suppressFwkInfo", {});
        topDesc.addUntracked<std::vector<std::string>>("suppressWarning", {});
        topDesc.addUntracked<std::vector<std::string>>("suppressError", {});
        topDesc.addUntracked<std::vector<std::string>>("debugModules", {});

        edm::ParameterSetDescription category;
        category.addUntracked<int>("reportEvery", 1);
        category.addUntracked<int>("limit", ThreadSafeLogMessageLoggerScribe::ConfigurableDefaults::NO_VALUE_SET)
            ->setComment(
                "Set a limit on the number of messages of this category. The default value is used to denote no "
                "limit.");
        category.addOptionalUntracked<int>("timespan");

        edm::ParameterSetDescription destination_base;
        destination_base.addOptionalUntracked<bool>("noLineBreaks");
        destination_base.addOptionalUntracked<bool>("noTimeStamps");
        destination_base.addOptionalUntracked<int>("lineLength");
        destination_base.addOptionalUntracked<std::string>("threshold");
        destination_base.addOptionalUntracked<std::string>("statisticsThreshold");

        edm::ParameterWildcard<edm::ParameterSetDescription> catnode("*", edm::RequireZeroOrMore, false, category);
        catnode.setComment("Specialize either a category or any of 'DEBUG', 'INFO', 'FWKINFO', 'WARNING' or 'ERROR'");
        destination_base.addNode(catnode);

        edm::ParameterSetDescription destination_noStats(destination_base);
        destination_noStats.addUntracked<bool>("enableStatistics", false);
        destination_noStats.addUntracked<bool>("resetStatistics", false);

        {
          edm::ParameterSetDescription default_pset;
          default_pset.addUntracked<int>("reportEvery", 1);
          default_pset.addUntracked<int>("limit", ThreadSafeLogMessageLoggerScribe::ConfigurableDefaults::NO_VALUE_SET)
              ->setComment(
                  "Set a limit on the number of messages of this category. The default value is used to denote no "
                  "limit.");
          default_pset.addOptionalUntracked<int>("timespan");
          default_pset.addUntracked<bool>("noLineBreaks", false);
          default_pset.addUntracked<bool>("noTimeStamps", false);
          default_pset.addUntracked<int>("lineLength", 80);
          default_pset.addUntracked<std::string>("threshold", "INFO");
          default_pset.addUntracked<std::string>("statisticsThreshold", "INFO");
          default_pset.addNode(catnode);

          edm::ParameterSetDescription cerr_destination(destination_base);
          cerr_destination.addUntracked<bool>("enableStatistics", true);
          cerr_destination.addUntracked<bool>("resetStatistics", false);
          cerr_destination.addUntracked<bool>("enable", true);

          edm::ParameterSetDescription cout_destination(destination_noStats);
          cout_destination.addUntracked<bool>("enable", false);

          edm::ParameterSetDescription fileDestination(destination_noStats);

          fileDestination.addOptionalUntracked<std::string>("output");
          fileDestination.addOptionalUntracked<std::string>("filename");
          fileDestination.addOptionalUntracked<std::string>("extension");
          edm::ParameterSetDescription files;
          edm::ParameterWildcard<edm::ParameterSetDescription> fileWildcard(
              "*", edm::RequireZeroOrMore, false, fileDestination);
          files.addNode(fileWildcard);

          std::map<std::string, edm::ParameterSetDescription> standards = {
              {"cerr", cerr_destination}, {"cout", cout_destination}, {"default", default_pset}, {"files", files}};

          edm::ParameterWildcardWithSpecifics psets("*", edm::RequireZeroOrMore, false, category, std::move(standards));
          topDesc.addNode(psets);
        }

        config.addDefault(topDesc);
      }
    }  // namespace

    void ThreadSafeLogMessageLoggerScribe::validate(edm::ParameterSet& pset) const {
      // See if old config API is being used
      if (usingOldConfig(pset))
        return;
      if (not pset.exists("files") and
          ((pset.exists("destinations") or pset.existsAs<std::vector<std::string>>("statistics", true) or
            pset.existsAs<std::vector<std::string>>("statistics", false) or pset.exists("categories")))) {
        return;
      }

      ConfigurationDescriptions config{"MessageLogger", "MessageLogger"};
      fillDescriptions(config);

      config.validate(pset, "MessageLogger");
    }

  }  // end of namespace service
}  // end of namespace edm
