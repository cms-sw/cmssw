// -*- C++ -*-
//
// Package:    TauJetCorrectorExample
// Class:      TauJetCorrectorExample
//
/**\class TauJetCorrectorExample TauJetCorrectorExample.cc MyTauAndHLTAnalyzer/TauJetCorrectorExample/src/TauJetCorrectorExample.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// necessary objects:
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include <DataFormats/Common/interface/RefVector.h>
#include <DataFormats/Common/interface/RefVectorIterator.h>
#include <DataFormats/Common/interface/Ref.h>
// basic cpp:

// basic cpp:
#include <iostream>
#include <string>
// some root includes
#include <TFile.h>
#include <TH1D.h>
#include <TMath.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TVector3.h>

//
// class decleration
//
class TauJetCorrectorExample : public edm::one::EDAnalyzer<> {
public:
  explicit TauJetCorrectorExample(const edm::ParameterSet&);
  ~TauJetCorrectorExample() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  std::string jetname;
  edm::EDGetTokenT<reco::IsolatedTauTagInfoCollection> tautoken;
  edm::EDGetTokenT<reco::JetCorrector> tauCorrectortoken;

  int nEvt;  // used to count the number of events
  int njets;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

using namespace reco;

TauJetCorrectorExample::TauJetCorrectorExample(const edm::ParameterSet& iConfig)
    : jetname(iConfig.getUntrackedParameter<std::string>("JetHandle", "iterativeCone5CaloJets")),
      tautoken(consumes<reco::IsolatedTauTagInfoCollection>(
          edm::InputTag(iConfig.getUntrackedParameter<std::string>("TauHandle", "coneIsolation")))),
      tauCorrectortoken(consumes<reco::JetCorrector>(
          iConfig.getUntrackedParameter<std::string>("tauCorrHandle", "TauJetCorrectorIcone5"))),
      nEvt(0),
      njets(0) {
  //now do what ever initialization is needed
}

TauJetCorrectorExample::~TauJetCorrectorExample() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TauJetCorrectorExample::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<reco::JetCorrector> taucorrector;
  iEvent.getByToken(tauCorrectortoken, taucorrector);

  // this analyzer produces a small root file with basic candidates and some MC information
  // some additional print statements
  ++nEvt;
  if ((nEvt % 10 == 0 && nEvt <= 100) || (nEvt % 100 == 0 && nEvt > 100))
    std::cout << "reading event " << nEvt << std::endl;

  // get taus
  Handle<reco::IsolatedTauTagInfoCollection> tauTagInfoHandle;
  iEvent.getByToken(tautoken, tauTagInfoHandle);
  reco::IsolatedTauTagInfoCollection::const_iterator tau = tauTagInfoHandle->begin();

  //  std::cout << "setting everything to 0 just before tau loop" << std::endl;
  njets = 0;

  std::cout << "starting tau loop" << std::endl;
  for (tau = tauTagInfoHandle->begin(); tau != tauTagInfoHandle->end() && njets < 10; ++tau) {
    //Should check tau discriminator, but not done here

    double pt = tau->jet().get()->et();

    //correction returns correction factor which must then be applied to original ET
    double scale = taucorrector->correction(tau->jet().get()->p4());
    double ptcorr = tau->jet().get()->et() * scale;

    std::cout << "Tau jet: Original Et = " << pt << " Corrected Et = " << ptcorr << std::endl;

    ++njets;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void
//TauJetCorrectorExample::beginJob(const edm::EventSetup& iSetup)
TauJetCorrectorExample::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void TauJetCorrectorExample::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(TauJetCorrectorExample);
