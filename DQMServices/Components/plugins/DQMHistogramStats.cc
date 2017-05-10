#include "DQMHistogramStats.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/MakerMacros.h"

namespace dqmservices {

DQMHistogramStats::DQMHistogramStats(edm::ParameterSet const & iConfig) 
    :workflow_ (""),
    child_ (""),
    producer_ ("DQM"),
    dirName_ ("."),
    fileBaseName_ (""),
    dumpOnEndLumi_(iConfig.getUntrackedParameter<bool>("dumpOnEndLumi", false)),
    dumpOnEndRun_(iConfig.getUntrackedParameter<bool>("dumpOnEndRun", false)),
    histogramNamesEndLumi_(iConfig.getUntrackedParameter<std::vector<std::string>>("histogramNamesEndLumi", std::vector<std::string>() )),
    histogramNamesEndRun_(iConfig.getUntrackedParameter<std::vector<std::string>>("histogramNamesEndRun", std::vector<std::string>() ))
    {
      std::copy( histogramNamesEndLumi_.begin(), histogramNamesEndLumi_.end(), std::inserter( histograms_, histograms_.end() ) );
      std::copy( histogramNamesEndRun_.begin(), histogramNamesEndRun_.end(), std::inserter( histograms_, histograms_.end() ) );
    }

DQMHistogramStats::~DQMHistogramStats() {}

//Set all possible x dimension parameters
void DQMHistogramStats::getDimensionX(Dimension &d, MonitorElement *m){
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

void DQMHistogramStats::getDimensionY(Dimension &d, MonitorElement *m){
  d.nBin = m->getNbinsY();
  d.low = m->getTH1()->GetYaxis()->GetXmin();
  d.up = m->getTH1()->GetYaxis()->GetXmax();
  d.mean = m->getTH1()->GetMean(2);
  d.meanError = m->getTH1()->GetMeanError(2);
  d.rms = m->getTH1()->GetRMS(2);
  d.rmsError = m->getTH1()->GetRMSError(2);
}

void DQMHistogramStats::getDimensionZ(Dimension &d, MonitorElement *m){
  d.nBin = m->getNbinsZ();
  d.low = m->getTH1()->GetZaxis()->GetXmin();
  d.up = m->getTH1()->GetZaxis()->GetXmax();
  d.mean = m->getTH1()->GetMean(3);
  d.meanError = m->getTH1()->GetMeanError(3);
  d.rms = m->getTH1()->GetRMS(3);
  d.rmsError = m->getTH1()->GetRMSError(3);
}

HistoEntry DQMHistogramStats::analyze(MonitorElement *m) {
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
      e.dimNumber = 2;
      break;
    case MonitorElement::DQM_KIND_TH2S:
      e.type = "TH2S";
      e.bin_size = sizeof(short);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      e.dimNumber = 2;
      break;
    case MonitorElement::DQM_KIND_TH2D:
      e.type = "TH2D";
      e.bin_size = sizeof(double);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      e.dimNumber = 2;
      break;
    case MonitorElement::DQM_KIND_TPROFILE2D:
      e.type = "TProfile2D";
      e.bin_size = sizeof(double);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      e.dimNumber = 2;
      break;

    // three-dim ME
    case MonitorElement::DQM_KIND_TH3F:
      e.type = "TH3F";
      e.bin_size = sizeof(float);
      getDimensionX(e.dimX, m);
      getDimensionY(e.dimY, m);
      getDimensionZ(e.dimZ, m);
      e.dimNumber = 3;
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

HistoStats DQMHistogramStats::collect(DQMStore::IGetter &iGetter, const std::set<std::string>& names ) {
  // new stat frame
  HistoStats st;

  if (names.size() > 0) {
    for (auto name: names) {
      std::cout << "***Getting MonitorElement " << name << " with IGetter" << std::endl;
      MonitorElement *m = iGetter.getElement(name); 
      auto frame = this->analyze(m); 
      st.insert(frame);
    };
    return st;
  }

  return st;
}

HistoStats DQMHistogramStats::collect(DQMStore::IGetter &iGetter, const std::vector<std::string>& names ) {
  // new stat frame
  HistoStats st;

  if (names.size() > 0) {
    for (auto name: names) {  
      MonitorElement *m = iGetter.getElement(name); 
      auto frame = this->analyze(m); 
      st.insert(frame);
    };
    return st;
  }

  return st;
}

HistoStats DQMHistogramStats::collect(DQMStore::IGetter &iGetter) {
  // new stat frame
  HistoStats st;

  auto collect_f = [&st, this](MonitorElement *m) -> void {
    auto frame = this->analyze(m);
    st.insert(frame);
  };
  std::cout << "***Getting values with IGetter" << std::endl;
  // correct stream id don't matter - no booking will be done
  iGetter.getAllContents_(collect_f, "");

  return st;
}

#if 0
void DQMHistogramStats::group(HistoStats& stats) {
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

std::string DQMHistogramStats::onlineOfflineFileName(const std::string &fileBaseName,
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

std::string DQMHistogramStats::getStepName(){
   std::ifstream comm("/proc/self/cmdline");
      std::string name;
      getline(comm, name);
      name = name.substr(name.find('\0', 0)+1);
      name = name.substr(0, name.find('\0', 0)-3);
      return name;
}

#if 0
std::unique_ptr<Stats> DQMHistogramStats::initializeGlobalCache(edm::ParameterSet const&) {
  return std::unique_ptr<Stats>(new Stats());
}

void DQMHistogramStats::analyze(edm::Event const&, edm::EventSetup const&) {
//This can safely be updated from multiple Streams because we are using an std::atomic
//      ++(globalCache()->value);
}

void DQMHistogramStats::globalEndJob(Stats const* iStats) {
  //std::cout <<"Number of events seen "<<iCount->value<<std::endl;
}
#endif

DEFINE_FWK_MODULE(DQMHistogramStats);

}  // end of namespace
