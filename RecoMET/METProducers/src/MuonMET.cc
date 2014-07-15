// -*- C++ -*-
//
// Package:    METProducers
// Class:      MuonMET
// 

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/MuonMET.h"

#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

//____________________________________________________________________________||
namespace cms 
{

//____________________________________________________________________________||
  MuonMET::MuonMET( const edm::ParameterSet& iConfig )
    : alg_()
    , metTypeInputTag_(iConfig.getParameter<edm::InputTag>("metTypeInputTag"))
  {

    edm::InputTag muonsInputTag = iConfig.getParameter<edm::InputTag>("muonsInputTag");
    inputMuonToken_ = consumes<edm::View<reco::Muon> >(muonsInputTag);

    edm::InputTag muonDepValueMap  = iConfig.getParameter<edm::InputTag>("muonMETDepositValueMapInputTag");
    inputValueMapMuonMetCorrToken_ = consumes<edm::ValueMap<reco::MuonMETCorrectionData> >(muonDepValueMap);

    edm::InputTag uncorMETInputTag = iConfig.getParameter<edm::InputTag>("uncorMETInputTag");
    if( metTypeInputTag_.label() == "CaloMET" )
      {
	inputCaloMETToken_ = consumes<edm::View<reco::CaloMET> >(uncorMETInputTag);
	produces<reco::CaloMETCollection>();
      }
    else
      {
	inputMETToken_ = consumes<edm::View<reco::MET> >(uncorMETInputTag);
	produces<reco::METCollection>();
      }
  }

  MuonMET::MuonMET() : alg_() {}

  void MuonMET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    edm::Handle<edm::View<reco::Muon> > inputMuons;
    iEvent.getByToken(inputMuonToken_, inputMuons);

    edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData> > vm_muCorrData_h;
    
    iEvent.getByToken(inputValueMapMuonMetCorrToken_, vm_muCorrData_h);
    
    if( metTypeInputTag_.label() == "CaloMET")
      {
	edm::Handle<edm::View<reco::CaloMET> > inputUncorMet;
	iEvent.getByToken(inputCaloMETToken_, inputUncorMet);
	std::auto_ptr<reco::CaloMETCollection> output( new reco::CaloMETCollection() );
	alg_.run(*(inputMuons.product()), *(vm_muCorrData_h.product()), *(inputUncorMet.product()), &*output);
	iEvent.put(output);
      }
    else
      {
	edm::Handle<edm::View<reco::MET> > inputUncorMet;
	iEvent.getByToken(inputMETToken_, inputUncorMet);
	std::auto_ptr<reco::METCollection> output( new reco::METCollection() );
	alg_.run(*(inputMuons.product()), *(vm_muCorrData_h.product()),*(inputUncorMet.product()), &*output);
	iEvent.put(output);
      }
  }
//____________________________________________________________________________||
  DEFINE_FWK_MODULE(MuonMET);

}

//____________________________________________________________________________||
