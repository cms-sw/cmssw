// -*- C++ -*-
//
// Package:    MuonMET
// Class:      MuonMET
// 
/**\class MuonMET MuonMET.cc JetMETCorrections/MuonMET/src/MuonMET.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Created:  Wed Aug 29 2007
//
//


// system include files
#include <memory>

#include <string.h>

// user include files
#include "MuonMET.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//using namespace std;

namespace cms 
{
  // PRODUCER CONSTRUCTORS ------------------------------------------
  MuonMET::MuonMET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    metType            = iConfig.getParameter<std::string>("metType");
   
    inputUncorMetLabel = iConfig.getParameter<edm::InputTag>("inputUncorMetLabel");
    inputMuonsLabel    = iConfig.getParameter<edm::InputTag>("inputMuonsLabel");
    muonPtMin      = iConfig.getParameter<double>("muonPtMin");
    muonEtaRange   = iConfig.getParameter<double>("muonEtaRange");
    muonTrackD0Max = iConfig.getParameter<double>("muonTrackD0Max");
    muonTrackDzMax = iConfig.getParameter<double>("muonTrackDzMax");
    muonNHitsMin   = iConfig.getParameter<int>("muonNHitsMin");
    muonDPtMax     = iConfig.getParameter<double>("muonDPtMax");
    muonChiSqMax   = iConfig.getParameter<double>("muonChiSqMax");
    muonDepositCor = iConfig.getParameter<bool>("muonDepositCor");
    if( metType == "CaloMET" )
      produces<CaloMETCollection>();
    else
      produces<METCollection>();

    edm::ParameterSet trackAssociatorParams = 
      iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
    trackAssociatorParameters_.loadParameters(trackAssociatorParams);   
    trackAssociator_.useDefaultPropagator();
  }
  MuonMET::MuonMET() : alg_() {}
  // PRODUCER DESTRUCTORS -------------------------------------------
  MuonMET::~MuonMET() {}

  // PRODUCER METHODS -----------------------------------------------
  void MuonMET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    using namespace edm;
    Handle<edm::View<reco::Muon> > inputMuons;
    iEvent.getByLabel( inputMuonsLabel, inputMuons );
    if( metType == "CaloMET")
      {
	Handle<CaloMETCollection> inputUncorMet;                     //Define Inputs
	iEvent.getByLabel( inputUncorMetLabel,  inputUncorMet );     //Get Inputs
	std::auto_ptr<CaloMETCollection> output( new CaloMETCollection() );  //Create empty output
	alg_.run( iEvent, iSetup,
		  *(inputUncorMet.product()), *(inputMuons.product()), 
		  muonPtMin, muonEtaRange,
		  muonTrackD0Max, muonTrackDzMax, 
		  muonNHitsMin, muonDPtMax, muonChiSqMax,
		  muonDepositCor, trackAssociator_, trackAssociatorParameters_,
		  &*output );  //Invoke the algorithm
	iEvent.put( output );                                        //Put output into Event
      }
    else
      {
	Handle<METCollection> inputUncorMet;                     //Define Inputs
	iEvent.getByLabel( inputUncorMetLabel,  inputUncorMet );     //Get Inputs
	std::auto_ptr<METCollection> output( new METCollection() );  //Create empty output
	alg_.run( iEvent, iSetup,
		  *(inputUncorMet.product()), *(inputMuons.product()), 
		  muonPtMin, muonEtaRange,
		  muonTrackD0Max, muonTrackDzMax, 
		  muonNHitsMin, muonDPtMax, muonChiSqMax,
		  muonDepositCor, trackAssociator_, trackAssociatorParameters_,
		  &*output );  //Invoke the algorithm
	iEvent.put( output );                                        //Put output into Event
      }
  }
  DEFINE_FWK_MODULE(MuonMET);
}//end namespace cms

