// Package:    TauMET
// Class:      TauMET
// 
// Original Authors:  Alfredo Gurrola, Chi Nhan Nguyen

#include "JetMETCorrections/Type1MET/interface/TauMET.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

using namespace reco;
using namespace std;
namespace cms 
{

  TauMET::TauMET(const edm::ParameterSet& iConfig) : _algo() {

    _InputTausLabel    = iConfig.getParameter<std::string>("InputTausLabel");
    _tauType    = iConfig.getParameter<std::string>("tauType");
    _InputCaloJetsLabel    = iConfig.getParameter<std::string>("InputCaloJetsLabel");
    _jetPTthreshold      = iConfig.getParameter<double>("jetPTthreshold");
    _jetEMfracLimit      = iConfig.getParameter<double>("jetEMfracLimit");
    _correctorLabel      = iConfig.getParameter<std::string>("correctorLabel");
    _InputMETLabel    = iConfig.getParameter<std::string>("InputMETLabel");
    _metType    = iConfig.getParameter<std::string>("metType");
    _JetMatchDeltaR      = iConfig.getParameter<double>("JetMatchDeltaR");
    _TauMinEt      = iConfig.getParameter<double>("TauMinEt");
    _TauEtaMax      = iConfig.getParameter<double>("TauEtaMax");
    _UseSeedTrack      = iConfig.getParameter<bool>("UseSeedTrack");
    _seedTrackPt      = iConfig.getParameter<double>("seedTrackPt");
    _UseTrackIsolation      = iConfig.getParameter<bool>("UseTrackIsolation");
    _trackIsolationMinPt      = iConfig.getParameter<double>("trackIsolationMinPt");
    _UseECALIsolation      = iConfig.getParameter<bool>("UseECALIsolation");
    _gammaIsolationMinPt      = iConfig.getParameter<double>("gammaIsolationMinPt");
    _UseProngStructure      = iConfig.getParameter<bool>("UseProngStructure");

    if( _metType == "recoMET" ) {
      produces< METCollection   >();
    } else {
      produces< CaloMETCollection   >();
    }

  }
  

  TauMET::~TauMET() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
  }
  
  // ------------ method called to produce the data  ------------
  void TauMET::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    using namespace edm;

    Handle<CaloJetCollection> calojetHandle;
    iEvent.getByLabel(_InputCaloJetsLabel, calojetHandle);
    const JetCorrector* correctedjets = JetCorrector::getJetCorrector (_correctorLabel, iSetup);

    Handle<PFTauCollection> tauHandle;
    iEvent.getByLabel(_InputTausLabel,tauHandle);

    if( _metType == "recoCaloMET" ) {
      Handle<CaloMETCollection> metHandle;
      iEvent.getByLabel(_InputMETLabel,metHandle);
      std::auto_ptr< CaloMETCollection > output( new CaloMETCollection() );
      _algo.run(iEvent,iSetup,tauHandle,calojetHandle,_jetPTthreshold,_jetEMfracLimit,*correctedjets,*(metHandle.product()),
                _JetMatchDeltaR,_TauMinEt,
                _TauEtaMax,_UseSeedTrack,_seedTrackPt,_UseTrackIsolation,_trackIsolationMinPt,_UseECALIsolation,
                _gammaIsolationMinPt,_UseProngStructure,&*output);
      iEvent.put( output );
    } else if( _metType == "recoMET" ) {
      Handle<METCollection> metHandle;
      iEvent.getByLabel(_InputMETLabel,metHandle);
      std::auto_ptr< METCollection > output( new METCollection() );
      _algo.run(iEvent,iSetup,tauHandle,calojetHandle,_jetPTthreshold,_jetEMfracLimit,*correctedjets,*(metHandle.product()),
                _JetMatchDeltaR,_TauMinEt,
                _TauEtaMax,_UseSeedTrack,_seedTrackPt,_UseTrackIsolation,_trackIsolationMinPt,_UseECALIsolation,
                _gammaIsolationMinPt,_UseProngStructure,&*output);
      iEvent.put( output );
    } else {
      std::cerr << "Incorrect Met Type!!! " << std::endl;
      std::cerr << "Please re-run and set the metType to 'recoCaloMET' or 'recoMET' " << std::endl;
      return;
    }

  }
  
  // ------------ method called once each job just before starting event loop  ------------
  void TauMET::beginJob() { }
  
  // ------------ method called once each job just after ending the event loop  ------------
  void TauMET::endJob() { }

  //DEFINE_FWK_MODULE(TauMET);  //define this as a plug-in
}

  
