#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/StreamerIO/plugins/DQMFileIterator.h"

#include <vector>
#include <map>
#include <sys/stat.h>

#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/range.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

namespace dqm {

class RamdiskMonitor : public DQMEDAnalyzer {
 public:
  RamdiskMonitor(const edm::ParameterSet &ps);
  virtual ~RamdiskMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

 protected:
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &,
                              edm::EventSetup const &) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const &lumi,
                                    edm::EventSetup const &eSetup) override;
  virtual void analyze(edm::Event const &e,
                       edm::EventSetup const &eSetup) override{};

  void analyzeFile(std::string fn, unsigned int run, unsigned int lumi,
                   std::string label);
  double getRunTimestamp();

  unsigned int runNumber_;
  std::string runInputDir_;
  std::vector<std::string> streamLabels_;
  std::string runPath_;

  struct StreamME {
    MonitorElement *eventsAccepted;
    MonitorElement *eventsProcessed;

    MonitorElement *deliveryDelayMTime;
    MonitorElement *deliveryDelayCTime;
  };

  std::map<std::string, StreamME> streams_;
  std::set<std::string> filesSeen_;
  std::chrono::high_resolution_clock::time_point runPathLastCollect_;
  double global_start_ = 0.;

  const double LUMI = 23.310893056;
};

RamdiskMonitor::RamdiskMonitor(const edm::ParameterSet &ps) {
  runNumber_ = ps.getUntrackedParameter<unsigned int>("runNumber");
  runInputDir_ = ps.getUntrackedParameter<std::string>("runInputDir");
  streamLabels_ =
      ps.getUntrackedParameter<std::vector<std::string> >("streamLabels");
};

RamdiskMonitor::~RamdiskMonitor(){};

void RamdiskMonitor::bookHistograms(DQMStore::IBooker &ib, edm::Run const &,
                                    edm::EventSetup const &) {
  runPath_ = str(boost::format("%s/run%06d") % runInputDir_ % runNumber_);

  for (auto stream : streamLabels_) {
    edm::LogInfo("RamdiskMonitor") << "Booking: " << stream;

    ib.cd();
    ib.setCurrentFolder(std::string("Info/RamdiskMonitor/") + stream + "/");

    StreamME m;

    m.eventsAccepted =
        ib.book1D("EventAccepted", "# of accepted events per lumi", 4, 0., 4.);
    m.eventsProcessed = ib.book1D("EventProcessed",
                                  "# of processed events per lumi", 4, 0., 4.);
    m.deliveryDelayMTime = ib.book1D(
        "DeliveryDelayMTime",
        "Observed delivery delay for the data file (mtime).", 4, 0., 4.);
    m.deliveryDelayCTime = ib.book1D(
        "DeliveryDelayCTime",
        "Observed delivery delay for the data file (ctime).", 4, 0., 4.);

    m.eventsAccepted->getTH1F()->SetCanExtend(TH1::kXaxis);
    m.eventsProcessed->getTH1F()->SetCanExtend(TH1::kXaxis);
    m.deliveryDelayMTime->getTH1F()->SetCanExtend(TH1::kXaxis);
    m.deliveryDelayCTime->getTH1F()->SetCanExtend(TH1::kXaxis);

    m.eventsAccepted->setAxisTitle("Luminosity Section", 1);
    m.eventsProcessed->setAxisTitle("Luminosity Section", 1);
    m.deliveryDelayMTime->setAxisTitle("Luminosity Section", 1);
    m.deliveryDelayCTime->setAxisTitle("Luminosity Section", 1);

    m.eventsAccepted->setAxisTitle("Number of events", 2);
    m.eventsProcessed->setAxisTitle("Number of events", 2);
    m.deliveryDelayMTime->setAxisTitle("Delay (s.)", 2);
    m.deliveryDelayCTime->setAxisTitle("Delay (s.)", 2);

    streams_[stream] = m;
  }
};

double RamdiskMonitor::getRunTimestamp() {
  if (global_start_ != 0) return global_start_;

  std::string run_global =
      str(boost::format("%s/.run%06d.global") % runInputDir_ % runNumber_);
  struct stat st;
  if (::stat(run_global.c_str(), &st) != 0) {
    edm::LogWarning("RamdiskMonitor") << "Stat failed: " << run_global;
    return 0.;
  }

  global_start_ = st.st_mtime;
  edm::LogPrint("RamdiskMonitor") << "Run start timestamp: " << global_start_;
  return global_start_;
};

void RamdiskMonitor::analyzeFile(std::string fn, unsigned int run,
                                 unsigned int lumi, std::string label) {
  using LumiEntry = dqmservices::DQMFileIterator::LumiEntry;

  // we are disabled, at least for this stream
  if (streams_.size() == 0) return;

  if (streams_.find(label) == streams_.end()) {
    edm::LogPrint("RamdiskMonitor") << "Stream not monitored [" << label
                                    << "]: " << fn;
    return;
  }

  StreamME m = streams_[label];

  // decode json and fill in some histograms
  LumiEntry lumi_jsn = LumiEntry::load_json(runPath_, fn, lumi, -1);
  m.eventsAccepted->setBinContent(lumi, lumi_jsn.n_events_accepted);
  m.eventsProcessed->setBinContent(lumi, lumi_jsn.n_events_processed);

  // collect stat struct and calculate mtimes
  struct stat st;
  if (::stat(fn.c_str(), &st) != 0) {
    edm::LogWarning("RamdiskMonitor") << "Stat failed: " << fn;
    return;
  }

  // get start offset (from .global)
  // abort the calculation if it does not exist
  double start_offset = getRunTimestamp();
  if (start_offset <= 0) return;

  // check fff_dqmtools (separate repository)
  // for calculation details
  double mtime = st.st_mtime;
  double ctime = st.st_ctime;

  // timeout from the begging of the run
  double start_offset_mtime = mtime - start_offset - LUMI;
  double start_offset_ctime = ctime - start_offset - LUMI;
  double lumi_offset = (lumi - 1) * LUMI;

  // timeout from the time we think this lumi happenned
  double delay_mtime = start_offset_mtime - lumi_offset;
  double delay_ctime = start_offset_ctime - lumi_offset;

  m.deliveryDelayMTime->setBinContent(lumi, delay_mtime);
  m.deliveryDelayCTime->setBinContent(lumi, delay_ctime);
};

void RamdiskMonitor::beginLuminosityBlock(edm::LuminosityBlock const &lumi,
                                          edm::EventSetup const &eSetup) {
  // search filesystem to find available lumi section files
  auto now = std::chrono::high_resolution_clock::now();
  runPathLastCollect_ = now;

  using boost::filesystem::directory_iterator;
  using boost::filesystem::directory_entry;

  directory_iterator dend;
  for (directory_iterator di(runPath_); di != dend; ++di) {
    const boost::regex fn_re("run(\\d+)_ls(\\d+)_([a-zA-Z0-9]+)(_.*)?\\.jsn");

    const std::string filename = di->path().filename().string();
    const std::string fn = di->path().string();

    if (filesSeen_.find(filename) != filesSeen_.end()) {
      continue;
    }

    boost::smatch result;
    if (boost::regex_match(filename, result, fn_re)) {
      unsigned int run = std::stoi(result[1]);
      unsigned int lumi = std::stoi(result[2]);
      std::string label = result[3];

      filesSeen_.insert(filename);

      if (run != runNumber_) continue;

      // check if this is EoR
      if ((lumi == 0) && (label == "EoR")) {
        // do not handle
        continue;
      }

      try {
        this->analyzeFile(fn, run, lumi, label);
      } catch (const std::exception &e) {
        // it's likely we have read it too soon
        filesSeen_.erase(filename);

        std::string msg("Found, tried to load the json, but failed (");
        msg += e.what();
        msg += "): ";
        edm::LogWarning("RamdiskMonitor") << msg;
      }
    }
  }

  // @TODO lookup info for the current lumi
};

void RamdiskMonitor::fillDescriptions(edm::ConfigurationDescriptions &d) {
  edm::ParameterSetDescription desc;

  desc.setComment(
      "Analyses file timestams in the /fff/ramdisk and creates monitor "
      "elements.");

  desc.addUntracked<std::vector<std::string> >("streamLabels")
      ->setComment("List of streams to monitor.");

  desc.addUntracked<unsigned int>("runNumber")
      ->setComment("Run number passed via configuration file.");

  desc.addUntracked<std::string>("runInputDir")
      ->setComment("Directory where the DQM files will appear.");

  d.add("RamdiskMonitor", desc);
}

}  // end of namespace

#include "FWCore/Framework/interface/MakerMacros.h"

typedef dqm::RamdiskMonitor RamdiskMonitor;
DEFINE_FWK_MODULE(RamdiskMonitor);
