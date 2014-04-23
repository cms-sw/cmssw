// -*- C++ -*-
//
// Package:    METProducers
// Class:      MuonMET
// 

//____________________________________________________________________________||
#ifndef RecoMET_MuonMET_h
#define RecoMET_MuonMET_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"

#include "RecoMET/METAlgorithms/interface/MuonMETAlgo.h"


//____________________________________________________________________________||
namespace cms 
{
  class MuonMET : public edm::stream::EDProducer<> 
  {
  public:
    explicit MuonMET( const edm::ParameterSet& );
    explicit MuonMET();
    virtual ~MuonMET() { }
    virtual void produce( edm::Event&, const edm::EventSetup& );

  private:
    MuonMETAlgo alg_;
    edm::InputTag metTypeInputTag_;

    edm::EDGetTokenT<edm::View<reco::Muon> > inputMuonToken_;
    edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > inputValueMapMuonMetCorrToken_;
    edm::EDGetTokenT<edm::View<reco::CaloMET> > inputCaloMETToken_;
    edm::EDGetTokenT<edm::View<reco::MET> > inputMETToken_;
    
  };
}

//____________________________________________________________________________||
#endif // RecoMET_MuonMET_h
