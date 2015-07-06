
#ifndef FWCore_Services_CondorStatusService_H_
#define FWCore_Services_CondorStatusService_H_

#include <atomic>
#include <string>

namespace edm {

    class ParameterSet;
    class ActivityRegistry;
    class StreamContext;
    class ConfigurationDescriptions;
    class GlobalContext;

    namespace service {

        class CondorStatusService
        {

        public:

            explicit CondorStatusService(ParameterSet const& pset, edm::ActivityRegistry& ar);
            ~CondorStatusService() {}
            CondorStatusService(const CondorStatusService&) = delete;
            CondorStatusService& operator=(const CondorStatusService&) = delete;

            void setUpdateInterval(unsigned int seconds) {m_updateInterval = seconds;}
            static void fillDescriptions(ConfigurationDescriptions &descriptions);

        private:

            bool isChirpSupported();
            bool updateChirp(std::string const &key, std::string const &value);
            inline void update();
            void forceUpdate();
            void updateImpl(time_t secsSinceLastUpdate);

            void eventPost(StreamContext const& iContext);
            void lumiPost(GlobalContext const&);
            void runPost(GlobalContext const&);
            void beginPost();
            void endPost();
            void filePost(std::string const &, bool);

            bool m_debug;
            static constexpr float m_defaultEmaInterval = 15*60; // Time in seconds to average EMA over for event rate.
            static constexpr unsigned int m_defaultUpdateInterval = 3*60;
            std::atomic<std::uint_least64_t> m_events;
            std::atomic<std::uint_least64_t> m_lumis;
            std::atomic<std::uint_least64_t> m_runs;
            std::atomic<std::uint_least64_t> m_files;
            std::atomic<time_t> m_lastUpdate;
            time_t m_beginJob = 0;
            std::atomic_flag m_shouldUpdate;
            time_t m_updateInterval = m_defaultUpdateInterval;
            float m_emaInterval = m_defaultEmaInterval;

            std::uint_least64_t m_lastEventCount = 0;
            float m_rate = 0;
        };

    }

}

#endif
