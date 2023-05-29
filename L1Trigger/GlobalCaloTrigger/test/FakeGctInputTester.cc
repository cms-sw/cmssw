// system include files
#include <memory>
#include <iostream>

// user include files

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "L1Trigger/GlobalCaloTrigger/test/FakeGctInputTester.h"

// Root includes
#include "TFile.h"
#include "TH1.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

//
// constructors and destructor
//
FakeGctInputTester::FakeGctInputTester(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed

  hFileName_ = iConfig.getUntrackedParameter<string>("histoFile", "FakeRctTest.root");
}

//
// member functions
//

// ------------ method called to for each event  ------------
void FakeGctInputTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<L1CaloEmCollection> rctCands;
  iEvent.getByLabel("fakeRct", rctCands);

  Handle<L1CaloRegionCollection> rctRgns;
  iEvent.getByLabel("fakeRct", rctRgns);

  Handle<L1GctEmCandCollection> gctIsoCands;
  iEvent.getByLabel("L1GctEmuDigis", "isoEm", gctIsoCands);

  Handle<L1GctEmCandCollection> gctNonIsoCands;
  iEvent.getByLabel("L1GctEmuDigis", "nonIsoEm", gctNonIsoCands);

  Handle<L1GctJetCandCollection> gctCJets;
  iEvent.getByLabel("L1GctEmuDigis", "cenJets", gctCJets);

  Handle<L1GctJetCandCollection> gctTJets;
  iEvent.getByLabel("L1GctEmuDigis", "tauJets", gctTJets);

  Handle<L1GctJetCandCollection> gctFJets;
  iEvent.getByLabel("L1GctEmuDigis", "forJets", gctFJets);

  // find non-zero RCT iso candidate
  int nRctIso = -1;
  for (unsigned i = 0; i < rctCands->size(); i++) {
    if (rctCands->at(i).isolated() && (rctCands->at(i).rank() > 0)) {
      nRctIso = i;
    }
  }

  if (nRctIso >= 0) {
    //     std::cout << "Found iso candidate at eta=" << rctCands->at(nRctIso).regionId().ieta() << " phi=" << rctCands->at(nRctIso).regionId().iphi() << std::endl;

    // does it match first GCT iso cand?
    unsigned int rctEta = rctCands->at(nRctIso).regionId().ieta();
    unsigned int gctEta = gctIsoCands->at(0).regionId().ieta();
    unsigned int rctPhi = rctCands->at(nRctIso).regionId().iphi();
    unsigned int gctPhi = gctIsoCands->at(0).regionId().iphi();

    isoEmDEta_->Fill(gctEta - rctEta);
    isoEmDPhi_->Fill(gctPhi - rctPhi);

    if ((rctEta != gctEta) || (rctPhi != gctPhi)) {
      cerr << "Iso EM mismatch" << endl;
      cerr << "RCT eta,phi : " << rctEta << "," << rctPhi << endl;
      cerr << "GCT eta,phi : " << gctEta << "," << gctPhi << endl;
    }
  }

  // find non-zero RCT non-iso candidate
  int nRctNonIso = -1;
  for (unsigned i = 0; i < rctCands->size(); i++) {
    if (!(rctCands->at(i).isolated()) && (rctCands->at(i).rank() > 0)) {
      nRctNonIso = i;
    }
  }

  if (nRctNonIso >= 0) {
    //     std::cout << "Found non-iso candidate at eta=" << rctCands->at(nRctNonIso).regionId().ieta() << " phi=" << rctCands->at(nRctNonIso).regionId().iphi() << std::endl;

    // does it match first GCT non-iso cand?
    unsigned int rctEta = rctCands->at(nRctNonIso).regionId().ieta();
    unsigned int gctEta = gctNonIsoCands->at(0).regionId().ieta();
    unsigned int rctPhi = rctCands->at(nRctNonIso).regionId().iphi();
    unsigned int gctPhi = gctNonIsoCands->at(0).regionId().iphi();

    nonIsoEmDEta_->Fill(gctEta - rctEta);
    nonIsoEmDPhi_->Fill(gctPhi - rctPhi);

    if ((rctEta != gctEta) || (rctPhi != gctPhi)) {
      cerr << "Noniso EM mismatch" << endl;
      cerr << "RCT eta,phi : " << rctEta << "," << rctPhi << endl;
      cerr << "GCT eta,phi : " << gctEta << "," << gctPhi << endl;
    }
  }

  // find non-zero RCT region
  int nRctRgn = -1;
  for (unsigned i = 0; i < rctRgns->size(); i++) {
    if (rctRgns->at(i).et() > 0) {
      nRctRgn = i;
    }
  }

  if (nRctRgn >= 0) {
    //     std::cout << "Found region at eta=" << rctCands->at(nRctRgn).regionId().ieta() << " phi=" << rctCands->at(nRctRgn).regionId().iphi() << std::endl;

    // does it match a  GCT jet?
    unsigned int rctEta = rctRgns->at(nRctRgn).id().ieta();
    unsigned int rctPhi = rctRgns->at(nRctRgn).id().iphi();
    unsigned int gctEta = 999;
    unsigned int gctPhi = 999;

    for (unsigned i = 0; i < gctCJets->size(); i++) {
      if (gctCJets->at(i).rank() > 0) {
        gctEta = gctCJets->at(i).regionId().ieta();
        gctPhi = gctCJets->at(i).regionId().iphi();
      }
    }
    for (unsigned i = 0; i < gctTJets->size(); i++) {
      if (gctTJets->at(i).rank() > 0) {
        gctEta = gctTJets->at(i).regionId().ieta();
        gctPhi = gctTJets->at(i).regionId().iphi();
      }
    }
    for (unsigned i = 0; i < gctFJets->size(); i++) {
      if (gctFJets->at(i).rank() > 0) {
        gctEta = gctFJets->at(i).regionId().ieta();
        gctPhi = gctFJets->at(i).regionId().iphi();
      }
    }

    jetDEta_->Fill(gctEta - rctEta);
    jetDPhi_->Fill(gctPhi - rctPhi);

    if ((rctEta != gctEta) || (rctPhi != gctPhi)) {
      cerr << "Region mismatch" << endl;
      cerr << "RCT eta,phi : " << rctEta << "," << rctPhi << endl;
      cerr << "GCT eta,phi : " << gctEta << "," << gctPhi << endl;
    }
  }

  // missing Et
}

// ------------ method called once each job just before starting event loop  ------------
void FakeGctInputTester::beginJob() {
  hFile_ = new TFile(hFileName_.c_str(), "RECREATE");

  isoEmDEta_ = new TH1F("isoEmDEta", "Iso EM delta eta", 41, -20.5, 20.5);
  isoEmDPhi_ = new TH1F("isoEmDPhi", "Iso EM delta phi", 41, -20, 20);

  nonIsoEmDEta_ = new TH1F("nonIsoEmDEta", "Non-iso EM delta eta", 41, -20.5, 20.5);
  nonIsoEmDPhi_ = new TH1F("nonIsoEmDPhi", "Non-iso EM delta phi", 41, -20.5, 20.5);

  jetDEta_ = new TH1F("jetDEta", "jet delta eta", 41, -20.5, 20.5);
  jetDPhi_ = new TH1F("jetDPhi", "jet delta phi", 41, -20.5, 20.5);
}

// ------------ method called once each job just after ending the event loop  ------------
void FakeGctInputTester::endJob() {
  hFile_->Write();
  hFile_->Close();
}
