#include "DQMHistogramJson.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/MakerMacros.h"

namespace dqmservices {

DQMHistogramJson::DQMHistogramJson(edm::ParameterSet const & iConfig) : DQMHistogramStats(iConfig){};

std::string DQMHistogramJson::toString(boost::property_tree::ptree doc)
{

    boost::regex exp("\"([0-9]+(\\.[0-9]+)?)\"");
    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, doc);
    std::string rv = boost::regex_replace(ss.str(), exp, "$1");

    return rv;
}

void DQMHistogramJson::writeMemoryJson(const std::string &fn, const HistoStats &stats) {
  using boost::property_tree::ptree;

  ptree doc;

  doc.put("pid", ::getpid());
  doc.put("n_entries", stats.size());
  doc.put("update_timestamp", std::time(NULL));

  ptree histograms;
  
  ptree info;
  ptree paths;
  ptree totals;
  for (auto &stat : stats) {
    ptree child;
    child.put("", stat.path);
    paths.push_back(std::make_pair("", child));
    child.put("", stat.total);
    totals.push_back(std::make_pair("", child));
  }

  info.add_child("path", paths);
  info.add_child("total", totals);
  histograms.push_back(std::make_pair("", info));

  doc.add_child("histograms", histograms);

  std::ofstream file(fn);
  file << toString(doc);
  file.close();
};

void DQMHistogramJson::dqmEndLuminosityBlock(DQMStore::IBooker &,
                                           DQMStore::IGetter &iGetter,
                                           edm::LuminosityBlock const &iLS,
                                           edm::EventSetup const &) {
  std::cout << "CALLED MY METHOD" << std::endl;
  if (dumpOnEndLumi_){
    std::cout << "CALLED MY METHOD INSIDE IF" << std::endl;
    HistoStats st = collect(iGetter);
    int irun     = iLS.id().run();
    int ilumi    = iLS.id().luminosityBlock();
    char suffix[64];
    sprintf(suffix, "R%09dLS%09d", irun, ilumi);
    workflow_ = "Default";
    dirName_ = getStepName();
    fileBaseName_ = dirName_ + "_" + producer_;// + version;
    std::string fileName = onlineOfflineFileName(fileBaseName_, std::string(suffix), workflow_, child_);
    writeMemoryJson(fileName, st);
  }
}

void DQMHistogramJson::dqmEndRun(DQMStore::IBooker &, 
                            DQMStore::IGetter &iGetter,
                            edm::Run const &iRun, 
                            edm::EventSetup const&) {
  if (dumpOnEndRun_){
    HistoStats st = collect(iGetter);
    int irun     = iRun.run();
    char suffix[64];
    sprintf(suffix, "R%09d", irun);
    workflow_ = "Default";
    dirName_ = getStepName();
    fileBaseName_ = dirName_ + "_" + producer_;// + version;
    std::string fileName = onlineOfflineFileName(fileBaseName_, std::string(suffix), workflow_, child_);
    writeMemoryJson(fileName, st);
  }
}

DEFINE_FWK_MODULE(DQMHistogramJson);

}  // end of namespace


