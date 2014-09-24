#include "DQMMonitoringService.h"

#include <boost/algorithm/string.hpp>

#include <ctime>

/*
 * This service is very similar to the FastMonitoringService in the HLT,
 * except that it is used for monitoring online DQM applications
 */

namespace dqmservices {

DQMMonitoringService::DQMMonitoringService(const edm::ParameterSet &pset, edm::ActivityRegistry& ar) {
  const char* x = getenv("MONDOG_PIPE");
  if (x) {
    std::cerr << "Monitoring pipe: " << x << std::endl;
    mstream_.reset(new std::ofstream(x));
  } else {
    std::cerr << "Monitoring fd not found, disabling." << std::endl;
  }

  nevents_ = 0;
  last_report_nevents_ = 0;
  last_report_time_ = std::chrono::high_resolution_clock::now();


  ar.watchPreGlobalBeginLumi(this, &DQMMonitoringService::evLumi);
  ar.watchPreSourceEvent(this, &DQMMonitoringService::evEvent);
}

DQMMonitoringService::~DQMMonitoringService() {
}

void DQMMonitoringService::update(std::function<void(ptree&)> f) {
  f(doc_);
}

void DQMMonitoringService::evLumi(GlobalContext const& iContext) {
  unsigned int run = iContext.luminosityBlockID().run();
  unsigned int lumi = iContext.luminosityBlockID().luminosityBlock();

  doc_.put("cmssw_run", run);
  doc_.put("cmssw_lumi", lumi);

  makeReport();
}

void DQMMonitoringService::evEvent(StreamID const& iContext) {
  nevents_ += 1;
  keepAlive();
}

void DQMMonitoringService::makeReport() {
  if (!mstream_)
    return;

  try {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::seconds;

    auto now = std::chrono::high_resolution_clock::now();
    float rate = (nevents_ - last_report_nevents_);
    rate = rate / duration_cast<seconds>(now - last_report_time_).count();

    doc_.put("events_total", nevents_);
    doc_.put("events_rate", rate);
    doc_.put("cmsRun_timestamp", std::time(NULL));

    write_json(*mstream_, doc_, false);
    mstream_->flush();

    last_report_time_ = now; 
    last_report_nevents_ = nevents_;
  } catch (...) {
    // pass
  }
}

void DQMMonitoringService::keepAlive() {
  if (!mstream_)
    return;

  *mstream_ << "\n";
  mstream_->flush();
}


} // end-of-namespace

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using dqmservices::DQMMonitoringService;
DEFINE_FWK_SERVICE(DQMMonitoringService);
