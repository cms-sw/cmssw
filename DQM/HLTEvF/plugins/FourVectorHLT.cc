// -*- C++ -*-
//
// Package:    FourVectorHLT
// Class:      FourVectorHLT
//
/**\class FourVectorHLT FourVectorHLT.cc DQM/FourVectorHLT/src/FourVectorHLT.cc

 Description: This is a DQM source meant to plot high-level HLT trigger
 quantities as stored in the HLT results object TriggerResults

*/
//
// Original Author:  Peter Wittich
//         Created:  May 2008
//
// Modernized by:    Marco Musich
//                   Dec 2025
//

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// system include files
#include <set>

class FourVectorHLT : public DQMEDAnalyzer {
public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  explicit FourVectorHLT(const edm::ParameterSet&);
  ~FourVectorHLT() override = default;

private:
  // DQM hooks
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // helper class to store the data
  class PathInfo {
    PathInfo() : pathIndex_(-1), pathName_("unset"), objectType_(-1) {}

  public:
    void setHistos(MonitorElement* const et,
                   MonitorElement* const eta,
                   MonitorElement* const phi,
                   MonitorElement* const etavsphi) {
      et_ = et;
      eta_ = eta;
      phi_ = phi;
      etavsphi_ = etavsphi;
    }
    MonitorElement* getEtHisto() { return et_; }
    MonitorElement* getEtaHisto() { return eta_; }
    MonitorElement* getPhiHisto() { return phi_; }
    MonitorElement* getEtaVsPhiHisto() { return etavsphi_; }
    const std::string getName(void) const { return pathName_; }
    ~PathInfo() {}
    PathInfo(std::string pathName, size_t type, float ptmin, float ptmax)
        : pathName_(pathName),
          objectType_(type),
          et_(nullptr),
          eta_(nullptr),
          phi_(nullptr),
          etavsphi_(nullptr),
          ptmin_(ptmin),
          ptmax_(ptmax) {}
    PathInfo(std::string pathName,
             size_t type,
             MonitorElement* et,
             MonitorElement* eta,
             MonitorElement* phi,
             MonitorElement* etavsphi,
             float ptmin,
             float ptmax)
        : pathName_(pathName),
          objectType_(type),
          et_(et),
          eta_(eta),
          phi_(phi),
          etavsphi_(etavsphi),
          ptmin_(ptmin),
          ptmax_(ptmax) {}
    bool operator==(const std::string v) { return v == pathName_; }

  private:
    int pathIndex_;
    std::string pathName_;
    int objectType_;

    // we don't own this data
    MonitorElement *et_, *eta_, *phi_, *etavsphi_;

    float ptmin_, ptmax_;

    const int index() { return pathIndex_; }

  public:
    const int type() { return objectType_; }
    float getPtMin() const { return ptmin_; }
    float getPtMax() const { return ptmax_; }
  };

  // simple collection - just
  class PathInfoCollection : public std::vector<PathInfo> {
  public:
    PathInfoCollection() : std::vector<PathInfo>() {}
    std::vector<PathInfo>::iterator find(std::string pathName) { return std::find(begin(), end(), pathName); }
  };

  // configuration

  const bool debug_;
  const bool plotAll_;
  const unsigned int nBins_;
  const double ptMin_, ptMax_;
  const std::string dirname_;
  const edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryToken_;

  // state
  PathInfoCollection hltPaths_;
  bool needRebook_{false};

  // For plotAll mode:
  std::set<std::string> pendingNewFilters_;
};

FourVectorHLT::FourVectorHLT(const edm::ParameterSet& iConfig)
    : debug_(iConfig.getUntrackedParameter<bool>("debug", false)),
      plotAll_(iConfig.getUntrackedParameter<bool>("plotAll", false)),
      nBins_(iConfig.getUntrackedParameter<unsigned int>("Nbins", 50)),
      ptMin_(iConfig.getUntrackedParameter<double>("ptMin", 0.)),
      ptMax_(iConfig.getUntrackedParameter<double>("ptMax", 200.)),
      dirname_(iConfig.getUntrackedParameter<std::string>("topFolderName", "HLT/FourVectorHLT")),
      triggerSummaryToken_(
          consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("triggerSummaryLabel"))) {
  // Predefined filters (static mode)
  auto filters = iConfig.getParameter<std::vector<edm::ParameterSet>>("filters");
  for (auto const& pset : filters) {
    hltPaths_.push_back(PathInfo(pset.getParameter<std::string>("name"),
                                 pset.getParameter<int>("type"),
                                 static_cast<float>(pset.getUntrackedParameter<double>("ptMin")),
                                 static_cast<float>(pset.getUntrackedParameter<double>("ptMax"))));
  }
}

void FourVectorHLT::dqmBeginRun(const edm::Run& theRun, const edm::EventSetup& theSetup) {
  if (!plotAll_)
    return;

  // Get TriggerEvent from first event **of this run**
  // We can't get an event here, so we mark that we will fill when receiving first event
  // In bookHistograms(), we don't know filter names yet. So:
  // Strategy:
  //   - in analyze(): if plotAll_, and beginRun found nothing yet, check filters once
  //   - add any missing filters to pendingNewFilters_
  //   - ask framework to call bookHistograms again (needRebook_ = true)

  pendingNewFilters_.clear();
  needRebook_ = false;  // will be set true in analyze() when we detect new filters
}

void FourVectorHLT::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(dirname_);

  // Book static mode:
  for (auto& p : hltPaths_) {
    if (!p.getEtHisto()) {
      std::string folder = dirname_ + "/" + p.getName();
      if (p.type() != 0) {
        folder += "/" + std::to_string(p.type());
      }
      ibooker.setCurrentFolder(folder);

      p.setHistos(ibooker.book1D(p.getName() + "_et", p.getName() + " E_{T}", nBins_, p.getPtMin(), p.getPtMax()),
                  ibooker.book1D(p.getName() + "_eta", p.getName() + " #eta", nBins_, -2.7, 2.7),
                  ibooker.book1D(p.getName() + "_phi", p.getName() + " #phi", nBins_, -3.14, 3.14),
                  ibooker.book2D(
                      p.getName() + "_etaphi", p.getName() + " #eta vs #phi", nBins_, -2.7, 2.7, nBins_, -3.14, 3.14));
    }
  }

  // Book filters discovered in this run
  for (auto const& name : pendingNewFilters_) {
    PathInfo p(name, 0, ptMin_, ptMax_);

    std::string folder = dirname_ + "/" + p.getName();
    if (p.type() != 0) {
      folder += "/" + std::to_string(p.type());
    }
    ibooker.setCurrentFolder(folder);

    p.setHistos(ibooker.book1D(name + "_et", name + " E_{T}", nBins_, 0, 100),
                ibooker.book1D(name + "_eta", name + " #eta", nBins_, -2.7, 2.7),
                ibooker.book1D(name + "_phi", name + " #phi", nBins_, -3.14, 3.14),
                ibooker.book2D(name + "_etaphi", name + " #eta vs #phi", nBins_, -2.7, 2.7, nBins_, -3.14, 3.14));

    hltPaths_.push_back(p);
  }

  pendingNewFilters_.clear();
  needRebook_ = false;
}

void FourVectorHLT::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<trigger::TriggerEvent> trig;
  iEvent.getByToken(triggerSummaryToken_, trig);
  if (!trig.isValid()) {
    edm::LogWarning("FourVectorHLT") << "TriggerEvent token is not valid, skipping event!";
    return;
  }

  const auto& toc = trig->getObjects();
  const auto& tags = trig->collectionTags();

  if (debug_) {
    edm::LogVerbatim("FourVectorHLT") << "process name: " << trig->usedProcessName();
    edm::LogVerbatim("FourVectorHLT") << "collection tags: ";
    for (const auto& tag : tags) {
      edm::LogVerbatim("FourVectorHLT") << "tag: " << tag;
    }

    edm::LogVerbatim("FourVectorHLT") << "toc size: " << toc.size();
    edm::LogVerbatim("FourVectorHLT") << "size filters: " << trig->sizeFilters();

    for (unsigned int i = 0; i < toc.size(); ++i) {
      trigger::TriggerObject const& triggerObject = toc[i];
      edm::LogVerbatim("FourVectorHLT") << "id:   " << triggerObject.id() << " pT :  " << triggerObject.pt()
                                        << " eta:  " << triggerObject.eta() << " phi:  " << triggerObject.phi()
                                        << " mass: " << triggerObject.mass();
    }

    for (size_t ia = 0; ia < trig->sizeFilters(); ++ia) {
      const auto& filterTagEncoded = trig->filterTagEncoded(ia);
      const auto& filterLabel = trig->filterLabel(ia);
      const auto& filterIndex = trig->filterIndex(trig->filterTag(ia));

      edm::LogVerbatim("FourVectorHLT") << "filter index:" << filterIndex << " filteTagEncoded: " << filterTagEncoded
                                        << " filterLabel: " << filterLabel;
      const auto& keys = trig->filterKeys(filterIndex);
      for (auto const& k : keys) {
        const auto& obj = toc[k];
        edm::LogVerbatim("FourVectorHLT") << "     name: " << filterLabel << " key: " << k << " id:" << obj.id()
                                          << "  pt: " << obj.pt() << " eta:" << obj.eta() << " phi:" << obj.phi();
      }
    }
  }

  // In plotAll_ mode, discover missing filters exactly once per run
  if (plotAll_) {
    if (!needRebook_) {
      for (size_t ia = 0; ia < trig->sizeFilters(); ++ia) {
        std::string fullname = trig->filterTag(ia).encode();
        std::string name = fullname.substr(0, fullname.find(':'));

        edm::LogInfo("FourVectorHLT") << "fullname: " << fullname << " name: " << name;

        bool known = false;
        for (auto const& p : hltPaths_) {
          if (p.getName() == name) {
            known = true;
            break;
          }
        }
        if (!known) {
          pendingNewFilters_.insert(name);
          needRebook_ = true;
        }
      }
    }

    // If new filters discovered, request rebooking now
    if (needRebook_) {
      // The framework will call bookHistograms() on next run transition
      return;  // skip filling until histos exist
    }
  }

  // Fill histograms
  for (auto& p : hltPaths_) {
    const auto& tagName = edm::InputTag(p.getName(), "", trig->usedProcessName());
    int index = trig->filterIndex(tagName);

    if (index >= trig->sizeFilters()) {
      if (debug_) {
        edm::LogWarning("FourVectorHLT") << "name: " << p.getName() << " index: " << index
                                         << " trig->sizeFilters(): " << trig->sizeFilters();
      }
      continue;
    }

    const auto& keys = trig->filterKeys(index);
    for (auto const& k : keys) {
      const auto& obj = toc[k];

      if (debug_) {
        edm::LogVerbatim("FourVectorHLT") << "name: " << p.getName() << " key: " << k << "  pt: " << obj.pt()
                                          << " eta:" << obj.eta() << " phi:" << obj.phi();
      }

      if (obj.id() == p.type()) {
        p.getEtHisto()->Fill(obj.pt());
        p.getEtaHisto()->Fill(obj.eta());
        p.getPhiHisto()->Fill(obj.phi());
        p.getEtaVsPhiHisto()->Fill(obj.eta(), obj.phi());
      } else {
        if ((std::abs(obj.id()) != std::abs(p.type())) && debug_)
          std::cout << "FourVectorHLT: "
                    << " name: " << p.getName() << " expected: " << obj.id() << " got: " << p.type() << std::endl;
      }
    }
  }
}

void FourVectorHLT::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("plotAll", false);
  desc.addUntracked<unsigned int>("Nbins", 50);
  desc.addUntracked<double>("ptMin", 0.);
  desc.addUntracked<double>("ptMax", 200.);

  edm::ParameterSetDescription filterDesc;
  filterDesc.add<std::string>("name", {});
  filterDesc.add<int>("type", 0);
  filterDesc.addUntracked<double>("ptMin", 0.);
  filterDesc.addUntracked<double>("ptMax", 200.);

  std::vector<edm::ParameterSet> default_toGet;
  desc.addVPSet("filters", filterDesc, default_toGet);

  desc.add<edm::InputTag>("triggerSummaryLabel", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));

  desc.addUntracked<std::string>("topFolderName", "HLT/FourVectorHLT");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FourVectorHLT);
