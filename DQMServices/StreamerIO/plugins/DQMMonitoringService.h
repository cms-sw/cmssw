#ifndef DQMServices_StreamerIO_DQMMonitoringService_h
#define DQMServices_StreamerIO_DQMMonitoringService_h

#include <chrono>

#include <boost/asio.hpp>
#include <boost/property_tree/ptree.hpp>

/*
 * This service is very similar to the FastMonitoringService in the HLT,
 * except that it is used for monitoring online DQM applications
 */

namespace edm {
  class ActivityRegistry;
  class GlobalContext;
  class ParameterSet;
  class StreamID;
}  // namespace edm

namespace dqmservices {

  class DQMMonitoringService {
  public:
    DQMMonitoringService(const edm::ParameterSet&, edm::ActivityRegistry&);
    ~DQMMonitoringService() = default;

    void connect();
    void keepAlive();

    void outputLumiUpdate();
    void outputUpdate(boost::property_tree::ptree& doc);

    void evLumi(edm::GlobalContext const&);
    void evEvent(edm::StreamID const&);

    void tryUpdate();

  private:
    boost::asio::local::stream_protocol::iostream mstream_;

    // global number of events processed
    long nevents_;

    // time point, number of events and lumi number at the time we switched to it
    std::chrono::high_resolution_clock::time_point last_lumi_time_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    long last_lumi_nevents_;
    // last lumi (we report stats for it, after we switch to the next one)
    unsigned long last_lumi_;

    unsigned long run_;   // current run
    unsigned long lumi_;  // current lumi
  };

}  // namespace dqmservices

#endif  // DQMServices_StreamerIO_DQMMonitoringService_h
