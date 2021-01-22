#ifndef FWCore_MessageService_ThreadSafeLogMessageLoggerScribe_h
#define FWCore_MessageService_ThreadSafeLogMessageLoggerScribe_h

#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "FWCore/MessageService/src/ELdestination.h"
#include "FWCore/MessageService/src/MessageLoggerDefaults.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

#include <iosfwd>
#include <vector>
#include <map>

#include <iostream>
#include <atomic>
#include "tbb/concurrent_queue.h"

namespace edm {
  namespace service {

    // ----------------------------------------------------------------------
    //
    // ThreadSafeLogMessageLoggerScribe.h
    //
    // OpCodeLOG_A_MESSAGE messages can be handled from multiple threads
    //
    // -----------------------------------------------------------------------

    class ELadministrator;
    class ELstatistics;

    class ThreadSafeLogMessageLoggerScribe : public AbstractMLscribe {
    public:
      // ---  birth/death:

      // ChangeLog 12
      /// --- If queue is NULL, this sets singleThread true
      explicit ThreadSafeLogMessageLoggerScribe();

      ~ThreadSafeLogMessageLoggerScribe() override;

      // --- receive and act on messages:
      // changelog 10
      void runCommand(MessageLoggerQ::OpCode opcode, void* operand) override;
      // changeLog 9

      struct ConfigurableDefaults {
        static constexpr int NO_VALUE_SET = -45654;
        static constexpr int COMMON_DEFAULT_LIMIT = NO_VALUE_SET;
        static constexpr int COMMON_DEFAULT_INTERVAL = NO_VALUE_SET;
        static constexpr int COMMON_DEFAULT_TIMESPAN = NO_VALUE_SET;

        std::string threshold_;
        int limit_;
        int reportEvery_;
        int timespan_;
        int lineLength_;
        bool noLineBreaks_;
        bool noTimeStamps_;
      };

    private:
      static ConfigurableDefaults parseDefaults(edm::ParameterSet const& job_pset);

      // --- convenience typedefs
      using vString = std::vector<std::string>;

      // --- log one consumed message
      void log(ErrorObj* errorobj_p);

      // --- cause statistics destinations to output
      void triggerStatisticsSummaries();
      void triggerFJRmessageSummary(std::map<std::string, double>& sm);

      // --- handle details of configuring via a ParameterSet:
      void configure_errorlog(edm::ParameterSet&);
      void configure_errorlog_new(edm::ParameterSet&);
      std::vector<std::string> configure_ordinary_destinations(edm::ParameterSet const&,
                                                               ConfigurableDefaults const& defaults,
                                                               vString const& categories);
      void configure_statistics(edm::ParameterSet const&,
                                ConfigurableDefaults const& defaults,
                                vString const& categories,
                                std::vector<std::string> const& destination_names);
      void configure_statistics_dest(edm::ParameterSet const& job_pset,
                                     ConfigurableDefaults const& defaults,
                                     vString const& categories,
                                     edm::ParameterSet const& stat_pset,
                                     std::string const& psetname,
                                     std::string const& filename);
      void configure_dest(edm::ParameterSet const& job_pset,
                          ConfigurableDefaults const&,
                          vString const& categories,
                          std::shared_ptr<ELdestination> dest_ctrl,
                          edm::ParameterSet const& dest_pset,
                          std::string const& filename);

      template <class T>
      static T getAparameter(edm::ParameterSet const& p, std::string const& id, T const& def) {
        T t = def;
        try {
          t = p.template getUntrackedParameter<T>(id, def);
        } catch (...) {
          try {
            t = p.template getParameter<T>(id);
          } catch (...) {
            // Since PSetValidation will catch such errors, we simply proceed as
            // best we can in case we are setting up the logger just to contain the
            // validation-caught error messages.
          }
        }
        return t;
      }

      // --- other helpers
      void parseCategories(std::string const& s, std::vector<std::string>& cats);
      std::string destinationFileName(edm::ParameterSet const&, std::string const&) const;
      std::shared_ptr<ELdestination> makeDestinationCtrl(std::string const& filename);

      void validate(edm::ParameterSet&) const;
      // --- data:
      edm::propagate_const<std::shared_ptr<ELadministrator>> m_admin_p;
      std::shared_ptr<ELdestination> m_early_dest;
      std::vector<edm::propagate_const<std::shared_ptr<std::ofstream>>> m_file_ps;
      std::map<std::string, edm::propagate_const<std::ostream*>> m_stream_ps;
      std::vector<std::shared_ptr<ELstatistics>> m_statisticsDestControls;
      std::vector<bool> m_statisticsResets;
      bool m_clean_slate_configuration;
      value_ptr<MessageLoggerDefaults> m_messageLoggerDefaults;
      bool m_active;
      std::atomic<bool> m_purge_mode;
      std::atomic<int> m_count;
      std::atomic<bool> m_messageBeingSent;
      tbb::concurrent_queue<ErrorObj*> m_waitingMessages;
      size_t m_waitingThreshold;
      std::atomic<unsigned long> m_tooManyWaitingMessagesCount;

    };  // ThreadSafeLogMessageLoggerScribe

  }  // end of namespace service
}  // namespace edm

#endif  // FWCore_MessageService_ThreadSafeLogMessageLoggerScribe_h
