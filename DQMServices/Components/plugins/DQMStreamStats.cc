#include "DQMStreamStats.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <chrono>
#include <ctime>

namespace dqmservices {

DQMStreamStats::DQMStreamStats(edm::ParameterSet const & iConfig) 
    :workflow_ (""),
    child_ (""),
    producer_ ("DQM"),
    dirName_ ("."),
    fileBaseName_ (""),
    dumpOnEndLumi_(iConfig.getUntrackedParameter<bool>("dumpOnEndLumi", false)),
    dumpOnEndRun_(iConfig.getUntrackedParameter<bool>("dumpOnEndRun", false))
    {}

DQMStreamStats::~DQMStreamStats() {}

//Set all possible x dimension parameters
void DQMStreamStats::getDimensionX(Dimension &d, MonitorElement *m){
  d.nBin = m->getNbinsX();
  d.low = m->getTH1()->GetXaxis()->GetXmin();
  d.up = m->getTH1()->GetXaxis()->GetXmax();
  d.mean = m->getTH1()->GetMean();
  d.meanError = m->getTH1()->GetMeanError();
  d.rms = m->getTH1()->GetRMS();
  d.rmsError = m->getTH1()->GetRMSError();
  d.underflow = m->getTH1()->GetBinContent(0);
  d.overflow = m->getTH1()->GetBinContent(d.nBin + 1);
}

void DQMStreamStats::getDimensionY(Dimension &d, MonitorElement *m){
  d.nBin = m->getNbinsY();
  d.low = m->getTH1()->GetYaxis()->GetXmin();
  d.up = m->getTH1()->GetYaxis()->GetXmax();
  d.mean = m->getTH1()->GetMean(2);
  d.meanError = m->getTH1()->GetMeanError(2);
  d.rms = m->getTH1()->GetRMS(2);
  d.rmsError = m->getTH1()->GetRMSError(2);
}

void DQMStreamStats::getDimensionZ(Dimension &d, MonitorElement *m){
  d.nBin = m->getNbinsZ();
  d.low = m->getTH1()->GetZaxis()->GetXmin();
  d.up = m->getTH1()->GetZaxis()->GetXmax();
  d.mean = m->getTH1()->GetMean(3);
  d.meanError = m->getTH1()->GetMeanError(3);
  d.rms = m->getTH1()->GetRMS(3);
  d.rmsError = m->getTH1()->GetRMSError(3);
}

HistoEntry DQMStreamStats::analyze(MonitorElement *m) {
  HistoEntry e;
  e.name = m->getName();
  e.path = m->getFullname();
  e.type = "unknown";

  switch (m->kind()) {
    case MonitorElement::DQM_KIND_INVALID:
      e.type = "INVALID";
      e.bin_size = -1;
      break;
    case MonitorElement::DQM_KIND_INT:
      e.type = "INT";
      e.bin_size = sizeof(int);
      break;
    case MonitorElement::DQM_KIND_REAL:
      e.type = "REAL";
      e.bin_size = sizeof(float);
      break;
    case MonitorElement::DQM_KIND_STRING:
      e.type = "STRING";
      e.bin_size = sizeof(char);
      break;

    // one-dim ME
    case MonitorElement::DQM_KIND_TH1F:
      e.type = "TH1F";
      e.bin_size = sizeof(float);
      getDimensionX(e.dimX, m);
      break;
    case MonitorElement::DQM_KIND_TH1S:
      e.type = "TH1S";
      e.bin_size = sizeof(short);
      getDimensionX(e.dimX, m);
      break;
    case MonitorElement::DQM_KIND_TH1D:
      e.type = "TH1D";
      e.bin_size = sizeof(double);
      getDimensionX(e.dimX, m);
      break;
    case MonitorElement::DQM_KIND_TPROFILE:
      e.type = "TProfile";
      e.bin_size = sizeof(double);
      getDimensionX(e.dimX, m);
      break;

    // two-dim ME
    case MonitorElement::DQM_KIND_TH2F:
      e.type = "TH2F";
      e.bin_size = sizeof(float);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      break;
    case MonitorElement::DQM_KIND_TH2S:
      e.type = "TH2S";
      e.bin_size = sizeof(short);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      break;
    case MonitorElement::DQM_KIND_TH2D:
      e.type = "TH2D";
      e.bin_size = sizeof(double);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      break;
    case MonitorElement::DQM_KIND_TPROFILE2D:
      e.type = "TProfile2D";
      e.bin_size = sizeof(double);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      break;

    // three-dim ME
    case MonitorElement::DQM_KIND_TH3F:
      e.type = "TH3F";
      e.bin_size = sizeof(float);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      getDimensionZ(e.dimZ, m);
      break;

    default:
      e.type = "unknown";
      e.bin_size = 0;
  };
  
  // skip "INVALID", "INT", "REAL", "STRING", "unknown"
  if (strcmp(e.type,"INVALID") && strcmp(e.type,"INT") && strcmp(e.type,"REAL") && strcmp(e.type,"STRING") && strcmp(e.type,"unknown")) {
      e.bin_count = m->getTH1()->GetNcells();
      e.entries = m->getEntries();
      e.maxBin = m->getTH1()->GetMaximumBin();
      e.minBin = m->getTH1()->GetMinimumBin();
      e.maxValue = m->getTH1()->GetMaximum();
      e.minValue = m->getTH1()->GetMinimum();
  }

  e.total = e.bin_count * e.bin_size + e.extra;

  return e;
}

HistoStats DQMStreamStats::collect(DQMStore::IGetter &iGetter) {
  // new stat frame
  HistoStats st;

  auto collect_f = [&st, this](MonitorElement *m) -> void {
    auto frame = this->analyze(m);
    st.insert(frame);
  };

  // correct stream id don't matter - no booking will be done
  iGetter.getAllContents_(collect_f, "");

  return st;
}

#if 0
void DQMStreamStats::group(HistoStats& stats) {
  HistoStats summaries;

  for (HistoEntry& entry : stats) {
    std::vector<std::string> tokens;
    boost::split(strs, entry.path, boost::is_any_of("/"));

    std::string prefix = entry.path;

    // for every non-base path component
    tokens.pop_back();
    for (std::string& token : tokens) {
      std::string summary_path = "unknown"
      
    }
  }
}
#endif

std::string DQMStreamStats::toString(boost::property_tree::ptree doc)
{

    boost::regex exp("\"([0-9]+(\\.[0-9]+)?)\"");
    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, doc);
    std::string rv = boost::regex_replace(ss.str(), exp, "$1");

    return rv;
}

void DQMStreamStats::writeMemoryJson(const std::string &fn, const HistoStats &stats) {
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

std::string DQMStreamStats::onlineOfflineFileName(const std::string &fileBaseName,
                      const std::string &suffix,
                      const std::string &workflow,
                      const std::string &child)
{
  size_t pos = 0;
  std::string wflow;
  wflow.reserve(workflow.size() + 3);
  wflow = workflow;
  while ((pos = wflow.find('/', pos)) != std::string::npos)
    wflow.replace(pos++, 1, "__");

  std::string filename = fileBaseName + suffix + wflow + child + ".json";
  return filename;
}

std::string DQMStreamStats::getStepName(){
   std::ifstream comm("/proc/self/cmdline");
      std::string name;
      getline(comm, name);
      name = name.substr(name.find('\0', 0)+1);
      name = name.substr(0, name.find('\0', 0)-3);
      return name;
}

void DQMStreamStats::dqmEndLuminosityBlock(DQMStore::IBooker &,
                                           DQMStore::IGetter &iGetter,
                                           edm::LuminosityBlock const &iLS,
                                           edm::EventSetup const &) {

  if (dumpOnEndLumi_){
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

void DQMStreamStats::dqmEndRun(DQMStore::IBooker &, 
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

#if 0
std::unique_ptr<Stats> DQMStreamStats::initializeGlobalCache(edm::ParameterSet const&) {
  return std::unique_ptr<Stats>(new Stats());
}

void DQMStreamStats::analyze(edm::Event const&, edm::EventSetup const&) {
//This can safely be updated from multiple Streams because we are using an std::atomic
//      ++(globalCache()->value);
}

void DQMStreamStats::globalEndJob(Stats const* iStats) {
  //std::cout <<"Number of events seen "<<iCount->value<<std::endl;
}
#endif

DEFINE_FWK_MODULE(DQMStreamStats);

}  // end of namespace


