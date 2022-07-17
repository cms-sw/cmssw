// -*- C++ -*-//
// Package:    Hcal
// Class:      HcalCollapseAnalyzer
//
/**\class HcalCollapseAnalyzer HcalCollapseAnalyzer.cc
 DQMOffline/Hcal/src/HcalCollapseAnalyzer.cc

 Description: Studies the collapsing of HcalRecHits

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu Dec 26 18:52:02 CST 2017
//
//

// system include files
#include <string>
#include <vector>

// Root objects
#include "TH1.h"

// user include files
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalCollapseAnalyzer : public DQMEDAnalyzer {
public:
  explicit HcalCollapseAnalyzer(const edm::ParameterSet &);
  ~HcalCollapseAnalyzer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  // ----------member data ---------------------------
  const std::string topFolderName_;
  const int verbosity_;
  const edm::InputTag recHitHBHE_, preRecHitHBHE_;
  const bool doHE_, doHB_;
  const HcalTopology *theHBHETopology = nullptr;

  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_prehbhe_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalTopologyToken_;

  MonitorElement *h_merge, *h_size, *h_depth;
  MonitorElement *h_sfrac, *h_frac, *h_balance;
};

HcalCollapseAnalyzer::HcalCollapseAnalyzer(const edm::ParameterSet &iConfig)
    : topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      recHitHBHE_(iConfig.getUntrackedParameter<edm::InputTag>("recHitHBHE", edm::InputTag("hbhereco"))),
      preRecHitHBHE_(iConfig.getUntrackedParameter<edm::InputTag>("preRecHitHBHE", edm::InputTag("hbheprereco"))),
      doHE_(iConfig.getUntrackedParameter<bool>("doHE", true)),
      doHB_(iConfig.getUntrackedParameter<bool>("doHB", false)),
      hcalTopologyToken_{esConsumes<HcalTopology, HcalRecNumberingRecord>()} {
  // define tokens for access
  tok_hbhe_ = consumes<HBHERecHitCollection>(recHitHBHE_);
  tok_prehbhe_ = consumes<HBHERecHitCollection>(preRecHitHBHE_);

  edm::LogVerbatim("Collapse") << "Verbosity " << verbosity_ << " with tags " << recHitHBHE_ << " and "
                               << preRecHitHBHE_ << " and Do " << doHB_ << ":" << doHE_;
}

void HcalCollapseAnalyzer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("topFolderName", "HcalCollapse");
  desc.addUntracked<int>("verbosity", 0);
  desc.addUntracked<edm::InputTag>("recHitHBHE", edm::InputTag("hbhereco"));
  desc.addUntracked<edm::InputTag>("preRecHitHBHE", edm::InputTag("hbheprereco"));
  desc.addUntracked<bool>("doHE", true);
  desc.addUntracked<bool>("doHB", false);
  descriptions.add("hcalCollapseAnalyzer", desc);
}

void HcalCollapseAnalyzer::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  if (verbosity_ > 0)
    edm::LogVerbatim("Collapse") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event()
                                 << " starts ==============";

  theHBHETopology = &iSetup.getData(hcalTopologyToken_);

  edm::Handle<HBHERecHitCollection> hbhereco;
  iEvent.getByToken(tok_hbhe_, hbhereco);
  edm::Handle<HBHERecHitCollection> hbheprereco;
  iEvent.getByToken(tok_prehbhe_, hbheprereco);
  if (verbosity_ > 0) {
    edm::LogVerbatim("Collapse") << "Handle Reco " << hbhereco << " Size " << hbhereco->size();
    edm::LogVerbatim("Collapse") << "Handle PreReco " << hbheprereco << " Size " << hbheprereco->size();
  }
  if (hbhereco.isValid() && hbheprereco.isValid()) {
    const HBHERecHitCollection *recohbhe = hbhereco.product();
    const HBHERecHitCollection *prerecohbhe = hbheprereco.product();
    if (verbosity_ > 0)
      edm::LogVerbatim("Collapse") << "Size of hbhereco: " << recohbhe->size()
                                   << " and hbheprereco: " << prerecohbhe->size();
    double sfrac = (prerecohbhe->empty()) ? 1 : ((double)(recohbhe->size())) / ((double)(prerecohbhe->size()));
    h_sfrac->Fill(sfrac);
    h_size->Fill(recohbhe->size());
    for (const auto &hit : (*recohbhe)) {
      HcalDetId id = hit.id();
      if (((id.subdet() == HcalEndcap) && doHE_) || ((id.subdet() == HcalBarrel) && doHB_)) {
        h_depth->Fill(id.depth());
        std::vector<HcalDetId> ids;
        theHBHETopology->unmergeDepthDetId(id, ids);
        if (verbosity_ > 0) {
          edm::LogVerbatim("Collapse") << id << " is derived from " << ids.size() << " components";
          for (unsigned int k = 0; k < ids.size(); ++k)
            edm::LogVerbatim("Collapse") << "[" << k << "] " << ids[k];
        }
        h_merge->Fill(ids.size());
        double energy = hit.energy();
        double etot(0);
        unsigned int k(0);
        for (const auto phit : (*prerecohbhe)) {
          if (std::find(ids.begin(), ids.end(), phit.id()) != ids.end()) {
            etot += phit.energy();
            double frac = (energy > 0) ? phit.energy() / energy : 1.0;
            h_frac->Fill(frac);
            if (verbosity_ > 0)
              edm::LogVerbatim("Collapse") << "Frac[" << k << "] " << frac << " for " << phit.id();
            ++k;
          }
        }
        double frac = (energy > 0) ? etot / energy : 1.0;
        h_balance->Fill(frac);
        if (verbosity_ > 0)
          edm::LogVerbatim("Collapse") << "Total energy " << energy << ":" << etot << ":" << frac;
      }
    }
  }
}

void HcalCollapseAnalyzer::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {
  // Book histograms
  ibooker.setCurrentFolder(topFolderName_);
  h_merge = ibooker.book1D("h_merge", "Number of hits merged", 10, 0.0, 10.0);
  h_size = ibooker.book1D("h_size", "Size of the RecHit collection", 100, 500.0, 1500.0);
  h_depth = ibooker.book1D("h_depth", "Depth of the Id's used", 10, 0.0, 10.0);
  h_sfrac = ibooker.book1D("h_sfrac", "Ratio of sizes of preRecHit and RecHit collections", 150, 0.0, 1.5);
  h_frac = ibooker.book1D("h_frac", "Fraction of energy before collapse", 150, 0.0, 1.5);
  h_balance = ibooker.book1D("h_balance", "Balance of energy between pre- and post-collapse", 100, 0.5, 1.5);
}

DEFINE_FWK_MODULE(HcalCollapseAnalyzer);
