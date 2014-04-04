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
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// necessary objects:
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
////#include "DataFormats/EgammaCandidates/interface/Electron.h"
////#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
////#include "DataFormats/MuonReco/interface/Muon.h"
////#include "DataFormats/JetReco/interface/GenJet.h"

#include "JetMETCorrections/TauJet/interface/TauJetCorrector.h"
//#include "JetMETCorrections/TauJet/interface/JetCalibratorTauJet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
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
class TauJetCorrectorExample : public edm::EDAnalyzer {
public:
  explicit TauJetCorrectorExample(const edm::ParameterSet&);
  ~TauJetCorrectorExample();


private:
  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  // ----------member data ---------------------------
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  edm::EDGetTokenT<ExampleData> exampletoken;
#endif
  std::string jetname;
  edm::EDGetTokenT<reco::IsolatedTauTagInfoCollection> tautoken;
  std::string tauCorrectorname;

  int nEvt;// used to count the number of events
  int njets;

  JetCorrector* taucorrector;

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

TauJetCorrectorExample::TauJetCorrectorExample(const edm::ParameterSet& iConfig):
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  exampletoken(consumes<ExampleData>(edm::InputTag("example"))),
#endif
  jetname(iConfig.getUntrackedParameter<std::string>("JetHandle","iterativeCone5CaloJets")),
  tautoken(consumes<reco::IsolatedTauTagInfoCollection>(edm::InputTag(iConfig.getUntrackedParameter<std::string>("TauHandle","coneIsolation")))),
  tauCorrectorname(iConfig.getUntrackedParameter<std::string>("tauCorrHandle", "TauJetCorrectorIcone5")),
  nEvt(0), njets(0), taucorrector(0)
{
  //now do what ever initialization is needed

}


TauJetCorrectorExample::~TauJetCorrectorExample()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TauJetCorrectorExample::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  taucorrector = const_cast<JetCorrector*>(JetCorrector::getJetCorrector(tauCorrectorname, iSetup));

  using namespace edm;

#ifdef THIS_IS_AN_EVENT_EXAMPLE
  Handle<ExampleData> pIn;
  iEvent.getByToken(exampletoken,pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
#endif
  using namespace edm;

  // this analyzer produces a small root file with basic candidates and some MC information
  // some additional print statements
  ++nEvt;
  if((nEvt%10==0 && nEvt<=100)||(nEvt%100==0 && nEvt>100))
    std::cout << "reading event " << nEvt << std::endl;

  // get taus
  Handle<reco::IsolatedTauTagInfoCollection> tauTagInfoHandle;
  iEvent.getByToken(tautoken,tauTagInfoHandle);
  reco::IsolatedTauTagInfoCollection::const_iterator tau=tauTagInfoHandle->begin();

  //get tau jet corrector
  //const JetCorrector* taucorrector = JetCorrector::getJetCorrector(tauCorrectorname, iSetup);

  //  std::cout << "setting everything to 0 just before tau loop" << std::endl;
  njets=0;

    std::cout << "starting tau loop" << std::endl;
    for(tau=tauTagInfoHandle->begin();tau!=tauTagInfoHandle->end() && njets<10;++tau) {

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
TauJetCorrectorExample::beginJob()
{

//	taucorrector = const_cast<JetCorrector*>(JetCorrector::getJetCorrector(tauCorrectorname, iSetup));

}

// ------------ method called once each job just after ending the event loop  ------------
void
TauJetCorrectorExample::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TauJetCorrectorExample);
