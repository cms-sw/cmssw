#include "DQMMonitoringService.h"

#include <boost/algorithm/string.hpp>

#include <ctime>

/*
 * This service is very similar to the FastMonitoringService in the HLT,
 * except that it is used for monitoring online DQM applications
 */

namespace dqmservices {

DQMMonitoringService::DQMMonitoringService(const edm::ParameterSet &pset, edm::ActivityRegistry& ar) {
  const char* x = getenv("DQMMON_UPDATE_PIPE");

  if (x) {
    std::cerr << "Monitoring pipe: " << x << std::endl;
    mstream_.reset(new std::ofstream(x));
  } else {
    std::cerr << "Monitoring file not found, disabling." << std::endl;
  }

  nevents_ = 0;
  last_report_nevents_ = 0;
  last_report_time_ = std::chrono::high_resolution_clock::now();

  ar.watchPreGlobalBeginLumi(this, &DQMMonitoringService::evLumi);
  ar.watchPreSourceEvent(this, &DQMMonitoringService::evEvent);
}

DQMMonitoringService::~DQMMonitoringService() {
}

void DQMMonitoringService::evLumi(GlobalContext const& iContext) {
  unsigned int run = iContext.luminosityBlockID().run();
  unsigned int lumi = iContext.luminosityBlockID().luminosityBlock();

  ptree doc;
  doc.put("cmssw_run", run);
  doc.put("cmssw_lumi", lumi);
  outputUpdate(doc);
}

void DQMMonitoringService::evEvent(StreamID const& iContext) {
  nevents_ += 1;

  using std::chrono::duration_cast;
  using std::chrono::seconds;
 
  auto now = std::chrono::high_resolution_clock::now();
  auto count = duration_cast<seconds>(now - last_report_time_).count();

  if (count < 30) {
    // we don't want to report too often
    return;
  }

  ptree doc;
  doc.put("events_total", nevents_);

  if (count > 0) {
    float rate = (nevents_ - last_report_nevents_) / count;
    doc.put("events_rate", rate);
  }

  last_report_time_ = now; 
  last_report_nevents_ = nevents_;

  outputUpdate(doc);
}

void DQMMonitoringService::outputUpdate(ptree& doc) {
  if (!mstream_)
    return;

  try {
    doc.put("update_timestamp", std::time(NULL));

    write_json(*mstream_, doc, false);
    mstream_->flush();
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
