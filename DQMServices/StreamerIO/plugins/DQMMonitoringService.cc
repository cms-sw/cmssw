#include "DQMMonitoringService.h"

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <ctime>
#include <iostream>

/*
 * This service is very similar to the FastMonitoringService in the HLT,
 * except that it is used for monitoring online DQM applications
 */

namespace dqmservices {

  DQMMonitoringService::DQMMonitoringService(const edm::ParameterSet& pset, edm::ActivityRegistry& ar) {
    const char* x = std::getenv("DQM2_SOCKET");
    if (x) {
      std::cerr << "Monitoring pipe: " << x << std::endl;
      mstream_.connect(boost::asio::local::stream_protocol::endpoint(x));
    } else {
      std::cerr << "Monitoring file not found, disabling." << std::endl;
    }

    // init counters
    nevents_ = 0;

    last_lumi_ = 0;
    last_lumi_nevents_ = 0;
    last_lumi_time_ = std::chrono::high_resolution_clock::now();

    run_ = 0;
    lumi_ = 0;

    ar.watchPreGlobalBeginLumi(this, &DQMMonitoringService::evLumi);
    ar.watchPreSourceEvent(this, &DQMMonitoringService::evEvent);
  }

  DQMMonitoringService::~DQMMonitoringService() {}

  void DQMMonitoringService::outputLumiUpdate() {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto now = std::chrono::high_resolution_clock::now();

    ptree doc;

    // these might be different than the numbers we want to report
    // rate/stats per lumi are calculated from last_*_ fields
    doc.put("cmssw_run", run_);
    doc.put("cmssw_lumi", lumi_);
    doc.put("events_total", nevents_);

    // do statistics for the last (elapsed) ls
    auto lumi_millis = duration_cast<milliseconds>(now - last_lumi_time_).count();
    auto lumi_events = nevents_ - last_lumi_nevents_;
    auto lumi_last = last_lumi_;

    if ((lumi_last > 0) && (lumi_millis > 0)) {
      doc.put("lumi_number", lumi_last);
      doc.put("lumi_events", lumi_events);
      doc.put("lumi_duration_ms", lumi_millis);

      float rate = ((float)lumi_events * 1000) / lumi_millis;
      doc.put("events_rate", rate);

      // also save the history entry
      ptree plumi;
      plumi.put("n", lumi_last);
      plumi.put("nevents", lumi_events);
      plumi.put("nmillis", lumi_millis);
      plumi.put("rate", rate);

      std::time_t hkey = std::time(nullptr);
      doc.add_child(str(boost::format("extra.lumi_stats.%d") % hkey), plumi);
    }

    outputUpdate(doc);
  }

  void DQMMonitoringService::evLumi(GlobalContext const& iContext) {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    // these might be different than the numbers we want to report
    // rate/stats per lumi are calculated from last_*_ fields
    run_ = iContext.luminosityBlockID().run();
    lumi_ = iContext.luminosityBlockID().luminosityBlock();

    outputLumiUpdate();

    last_lumi_time_ = std::chrono::high_resolution_clock::now();
    last_lumi_nevents_ = nevents_;
    last_lumi_ = lumi_;
  }

  void DQMMonitoringService::evEvent(StreamID const& iContext) {
    nevents_ += 1;
    tryUpdate();
  }

  void DQMMonitoringService::outputUpdate(ptree& doc) {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    if (!mstream_)
      return;

    try {
      last_update_time_ = std::chrono::high_resolution_clock::now();
      doc.put("update_timestamp", std::time(nullptr));

      write_json(mstream_, doc, false);
      mstream_.flush();
    } catch (...) {
      // pass
    }
  }

  void DQMMonitoringService::keepAlive() {
    if (!mstream_)
      return;

    mstream_ << "\n";
    mstream_.flush();

    tryUpdate();
  }

  void DQMMonitoringService::tryUpdate() {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    if (!mstream_)
      return;

    // sometimes we don't see any transition for a very long time
    // but we still want updates
    // luckily, keepAlive is called rather often by the input source
    auto now = std::chrono::high_resolution_clock::now();
    auto millis = duration_cast<milliseconds>(now - last_update_time_).count();
    if (millis >= (25 * 1000)) {
      outputLumiUpdate();
    }
  }

}  // namespace dqmservices

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using dqmservices::DQMMonitoringService;
DEFINE_FWK_SERVICE(DQMMonitoringService);
