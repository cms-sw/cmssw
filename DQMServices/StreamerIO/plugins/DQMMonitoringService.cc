#include "DQMMonitoringService.h"

/*
 * This service is very similar to the FastMonitoringService in the HLT,
 * except that it is used for monitoring online DQM applications
 */

namespace dqmservices {

namespace fs = boost::filesystem;

DQMMonitoringService::DQMMonitoringService(const edm::ParameterSet &pset, edm::ActivityRegistry&) {
  json_path_ = pset.getUntrackedParameter<std::string>("jsonPath");

  char host[128];
  if (gethostname(host ,sizeof(host)) == -1) {
    throw cms::Exception("DQMMonitoringService")
          << "Internal error, cannot get host name";
  }

  hostname_ = host;
  fseq_ = 0;
}

DQMMonitoringService::~DQMMonitoringService() {
}

void DQMMonitoringService::registerExtra(std::string name, ptree data) {
  extra_.add_child(name, data);
}

void DQMMonitoringService::reportLumiSection(int run, int lumi) {
  int pid = getpid();
  ++fseq_;

  if (! fs::is_directory(json_path_)) {
    extra_.clear();
    return; // no directory present, quit
  }

  // output jsn file
  std::string path =
      str(boost::format("dqm-source-state-run%06d-lumi%04d-id%d-seq%d.jsn") % 
        run % lumi % pid % fseq_);

  path = (json_path_ / path).string();

  using namespace boost::property_tree;
  ptree pt;

  pt.put("_id",
    str(boost::format("dqm-source-state-run%06d-pid%06d") % run % pid));
  pt.put("pid", pid);
  pt.put("tag", "not-implemented");
  pt.put("hostname", hostname_); 
  pt.put("sequence", fseq_);
  pt.put("type", "dqm-source-state");
  pt.put("run", run);
  pt.put("lumi", lumi);
  pt.add_child("extra", extra_);

  std::ofstream file(path);
  write_json(file, pt, true);
  file.close();

  extra_.clear();
}

} // end-of-namespace

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using dqmservices::DQMMonitoringService;
DEFINE_FWK_SERVICE(DQMMonitoringService);
