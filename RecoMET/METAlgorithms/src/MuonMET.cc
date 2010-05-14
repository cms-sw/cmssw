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
#include "RecoMET/METAlgorithms/interface/MuonMET.h"

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

#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//using namespace std;

namespace cms 
{
  // PRODUCER CONSTRUCTORS ------------------------------------------
  MuonMET::MuonMET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    metTypeInputTag_             = iConfig.getParameter<edm::InputTag>("metTypeInputTag");
    uncorMETInputTag_            = iConfig.getParameter<edm::InputTag>("uncorMETInputTag");
    muonsInputTag_               = iConfig.getParameter<edm::InputTag>("muonsInputTag");
    muonDepValueMap_             = iConfig.getParameter<edm::InputTag>("muonMETDepositValueMapInputTag");

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

    Handle<ValueMap<reco::MuonMETCorrectionData> > vm_muCorrData_h;
    
    iEvent.getByLabel( muonDepValueMap_, vm_muCorrData_h);
    
    if( metTypeInputTag_.label() == "CaloMET")
      {
	Handle<View<reco::CaloMET> > inputUncorMet;
	iEvent.getByLabel( uncorMETInputTag_, inputUncorMet  );     //Get Inputs
	std::auto_ptr<CaloMETCollection> output( new CaloMETCollection() );  //Create empty output
	
	alg_.run(*(inputMuons.product()), *(vm_muCorrData_h.product()),
		 *(inputUncorMet.product()), &*output);
	
	iEvent.put(output);                                        //Put output into Event
      }
    else
      {
	Handle<View<reco::MET> > inputUncorMet;                     //Define Inputs
	iEvent.getByLabel( uncorMETInputTag_,  inputUncorMet );     //Get Inputs
	std::auto_ptr<METCollection> output( new METCollection() );  //Create empty output
	

	alg_.run(*(inputMuons.product()), *(vm_muCorrData_h.product()),*(inputUncorMet.product()), &*output);
	iEvent.put(output);                                        //Put output into Event
      }
  }
}//end namespace cms

using cms::MuonMET;
DEFINE_FWK_MODULE(MuonMET);
