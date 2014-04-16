// -*- C++ -*-
//
// Package:    METProducers
// Class:      MuonMET
// 

//____________________________________________________________________________||
#include <memory>

#include <string.h>

#include "RecoMET/METProducers/interface/MuonMET.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/MakerMacros.h"

//____________________________________________________________________________||
namespace cms 
{
  MuonMET::MuonMET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    metTypeInputTag_             = iConfig.getParameter<edm::InputTag>("metTypeInputTag");
    uncorMETInputTag_            = iConfig.getParameter<edm::InputTag>("uncorMETInputTag");
    muonsInputTag_               = iConfig.getParameter<edm::InputTag>("muonsInputTag");
    muonDepValueMap_             = iConfig.getParameter<edm::InputTag>("muonMETDepositValueMapInputTag");

    inputMuonToken_ = consumes<edm::View<reco::Muon> >(muonsInputTag_);
    inputValueMapMuonMetCorrToken_ = consumes<edm::ValueMap<reco::MuonMETCorrectionData> >(muonDepValueMap_);

    if( metTypeInputTag_.label() == "CaloMET" ) {
      inputCaloMETToken_ = consumes<edm::View<reco::CaloMET> >(uncorMETInputTag_);
      produces<reco::CaloMETCollection>();
    } else 
      inputMETToken_ = consumes<edm::View<reco::MET> >(uncorMETInputTag_);
      produces<reco::METCollection>();
    
  }
  MuonMET::MuonMET() : alg_() {}

  MuonMET::~MuonMET() {}

  void MuonMET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    using namespace edm;
    
    Handle<View<reco::Muon> > inputMuons;
    iEvent.getByToken(inputMuonToken_, inputMuons );

    Handle<ValueMap<reco::MuonMETCorrectionData> > vm_muCorrData_h;
    
    iEvent.getByToken(inputValueMapMuonMetCorrToken_, vm_muCorrData_h);
    
    if( metTypeInputTag_.label() == "CaloMET")
      {
	Handle<View<reco::CaloMET> > inputUncorMet;
	iEvent.getByToken(inputCaloMETToken_, inputUncorMet);     //Get Inputs
	std::auto_ptr<reco::CaloMETCollection> output( new reco::CaloMETCollection() );  //Create empty output
	
	alg_.run(*(inputMuons.product()), *(vm_muCorrData_h.product()),
		 *(inputUncorMet.product()), &*output);
	
	iEvent.put(output);                                        //Put output into Event
      }
    else
      {
	Handle<View<reco::MET> > inputUncorMet;                     //Define Inputs
	iEvent.getByToken(inputMETToken_, inputUncorMet);     //Get Inputs
	std::auto_ptr<reco::METCollection> output( new reco::METCollection() );  //Create empty output
	

	alg_.run(*(inputMuons.product()), *(vm_muCorrData_h.product()),*(inputUncorMet.product()), &*output);
	iEvent.put(output);                                        //Put output into Event
      }
  }
}

//____________________________________________________________________________||
