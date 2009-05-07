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
#include "JetMETCorrections/Type1MET/interface/MuonMET.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

//using namespace std;

namespace cms 
{
  // PRODUCER CONSTRUCTORS ------------------------------------------
  MuonMET::MuonMET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    metTypeInputTag_     = iConfig.getParameter<edm::InputTag>("metTypeInputTag");
    uncorMETInputTag_    = iConfig.getParameter<edm::InputTag>("uncorMETInputTag");
    muonsInputTag_       = iConfig.getParameter<edm::InputTag>("muonsInputTag");
    useTrackAssociatorPositions_ = iConfig.getParameter<bool>("useTrackAssociatorPositions");
    useRecHits_          = iConfig.getParameter<bool>("useRecHits");
    useHO_               = iConfig.getParameter<bool>("useHO");
    towerEtThreshold_    = iConfig.getParameter<double>("towerEtThreshold");
 
    edm::ParameterSet trackAssociatorParams =
      iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
    trackAssociatorParameters_.loadParameters(trackAssociatorParams);
    trackAssociator_.useDefaultPropagator();

    if( metTypeInputTag_.label() == "CaloMET" ) {
      produces<CaloMETCollection>();
    } else 
      produces<METCollection>();
    
  }
  MuonMET::MuonMET() : alg_() {}
  // PRODUCER DESTRUCTORS -------------------------------------------
  MuonMET::~MuonMET() {}

  // PRODUCER METHODS -----------------------------------------------
  void MuonMET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    using namespace edm;
    
    //get the muons
    Handle<View<reco::Muon> > inputMuons;
    iEvent.getByLabel( muonsInputTag_, inputMuons );

    if( metTypeInputTag_.label() == "CaloMET")
      {
	Handle<View<reco::CaloMET> > inputUncorMet;
	iEvent.getByLabel( uncorMETInputTag_, inputUncorMet  );     //Get Inputs
	std::auto_ptr<CaloMETCollection> output( new CaloMETCollection() );  //Create empty output
	
	//new MET cor
	alg_.run(iEvent, iSetup, *(inputUncorMet.product()),
		 *(inputMuons.product()), 
		 trackAssociator_,
		 trackAssociatorParameters_,
		 &*output,
		 useTrackAssociatorPositions_,
                 useRecHits_, useHO_,
		 towerEtThreshold_);
	
	iEvent.put(output);                                        //Put output into Event
      }
    else
      {
	Handle<View<reco::MET> > inputUncorMet;                     //Define Inputs
	iEvent.getByLabel( uncorMETInputTag_,  inputUncorMet );     //Get Inputs
	std::auto_ptr<METCollection> output( new METCollection() );  //Create empty output
	
	alg_.run(iEvent, iSetup, *(inputUncorMet.product()),
		 *(inputMuons.product()), 
		 trackAssociator_,
		 trackAssociatorParameters_,
		 &*output,
		 useTrackAssociatorPositions_,
                 useRecHits_, useHO_,
		 towerEtThreshold_);
	
	iEvent.put( output );                                        //Put output into Event
      }
  }
}//end namespace cms

