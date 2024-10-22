#include <string>
#include <vector>
#include <map>

#include "DQM/TrackingMonitor/interface/GetLumi.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace {
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  struct MEbinning {
    int nbins;
    double xmin;
    double xmax;
  };

  struct Histograms {
    dqm::reco::MonitorElement* psColumnIndexVsLS;
  };
}  // namespace

//
// class declaration
//

class PSMonitor : public DQMGlobalEDAnalyzer<Histograms> {
public:
  PSMonitor(const edm::ParameterSet&);
  ~PSMonitor() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription& pset, int value);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;
  void dqmAnalyze(edm::Event const& event, edm::EventSetup const& setup, Histograms const&) const override;

private:
  void getHistoPSet(edm::ParameterSet& pset, MEbinning& mebinning);

  std::string folderName_;

  edm::EDGetTokenT<GlobalAlgBlkBxCollection> ugtBXToken_;

  MEbinning ps_binning_;
  MEbinning ls_binning_;
};

// -----------------------------
//  constructors and destructor
// -----------------------------

PSMonitor::PSMonitor(const edm::ParameterSet& config)
    : folderName_(config.getParameter<std::string>("folderName")),
      ugtBXToken_(consumes<GlobalAlgBlkBxCollection>(config.getParameter<edm::InputTag>("ugtBXInputTag"))) {
  edm::ParameterSet histoPSet = config.getParameter<edm::ParameterSet>("histoPSet");
  edm::ParameterSet psColumnPSet = histoPSet.getParameter<edm::ParameterSet>("psColumnPSet");
  edm::ParameterSet lsPSet = histoPSet.getParameter<edm::ParameterSet>("lsPSet");

  getHistoPSet(psColumnPSet, ps_binning_);
  getHistoPSet(lsPSet, ls_binning_);
}

void PSMonitor::getHistoPSet(edm::ParameterSet& pset, MEbinning& mebinning) {
  mebinning.nbins = pset.getParameter<int32_t>("nbins");
  mebinning.xmin = 0.;
  mebinning.xmax = double(pset.getParameter<int32_t>("nbins"));
}

void PSMonitor::bookHistograms(DQMStore::IBooker& booker,
                               edm::Run const& run,
                               edm::EventSetup const& setup,
                               Histograms& histograms) const {
  std::string histname, histtitle;

  std::string currentFolder = folderName_;
  booker.setCurrentFolder(currentFolder);

  int nbins;
  double xmin, xmax;
  std::vector<std::string> labels;
  edm::Service<edm::service::PrescaleService> prescaleService;
  if (prescaleService.isAvailable() and not prescaleService->getLvl1Labels().empty()) {
    labels = prescaleService->getLvl1Labels();
    nbins = labels.size();
    xmin = 0.;
    xmax = double(labels.size());
  } else {
    nbins = ps_binning_.nbins;
    xmin = ps_binning_.xmin;
    xmax = ps_binning_.xmax;
    labels.resize(nbins, "");
  }

  histname = "psColumnIndexVsLS";
  histtitle = "PS column index vs LS";
  auto me =
      booker.book2D(histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, nbins, xmin, xmax);
  me->setAxisTitle("LS", 1);
  me->setAxisTitle("PS column index", 2);
  histograms.psColumnIndexVsLS = me;

  int bin = 1;
  for (auto const& l : labels) {
    histograms.psColumnIndexVsLS->setBinLabel(bin, l, 2);
    bin++;
  }
}

void PSMonitor::dqmAnalyze(edm::Event const& event, edm::EventSetup const& setup, Histograms const& histograms) const {
  int ls = event.id().luminosityBlock();
  int psColumn = -1;

  edm::Handle<GlobalAlgBlkBxCollection> ugtBXhandle;
  event.getByToken(ugtBXToken_, ugtBXhandle);
  if (ugtBXhandle.isValid() and not ugtBXhandle->isEmpty(0)) {
    psColumn = ugtBXhandle->at(0, 0).getPreScColumn();
  }
  histograms.psColumnIndexVsLS->Fill(ls, psColumn);
}

void PSMonitor::fillHistoPSetDescription(edm::ParameterSetDescription& pset, int value) {
  pset.add<int>("nbins", value);
}

void PSMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ugtBXInputTag", edm::InputTag("hltGtStage2Digis"));
  desc.add<std::string>("folderName", "HLT/PSMonitoring");

  edm::ParameterSetDescription histoPSet;

  edm::ParameterSetDescription psColumnPSet;
  fillHistoPSetDescription(psColumnPSet, 8);
  histoPSet.add("psColumnPSet", psColumnPSet);

  edm::ParameterSetDescription lsPSet;
  fillHistoPSetDescription(lsPSet, 2500);
  histoPSet.add("lsPSet", lsPSet);

  desc.add("histoPSet", histoPSet);

  descriptions.add("psMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PSMonitor);
