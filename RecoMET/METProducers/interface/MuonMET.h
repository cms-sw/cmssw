// -*- C++ -*-
//
// Package:    METProducers
// Class:      MuonMET
// 

//____________________________________________________________________________||
#ifndef RecoMET_MuonMET_h
#define RecoMET_MuonMET_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "RecoMET/METAlgorithms/interface/MuonMETAlgo.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"


//____________________________________________________________________________||
namespace cms 
{
  class MuonMET : public edm::EDProducer 
  {
  public:
    explicit MuonMET( const edm::ParameterSet& );
    explicit MuonMET();
    virtual ~MuonMET();
    virtual void produce( edm::Event&, const edm::EventSetup& );

  private:
    MuonMETAlgo alg_;
    edm::InputTag metTypeInputTag_;
    edm::InputTag uncorMETInputTag_;
    edm::InputTag muonsInputTag_;
    edm::InputTag muonDepValueMap_;

    edm::EDGetTokenT<edm::View<reco::Muon> > inputMuonToken_;
    edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > inputValueMapMuonMetCorrToken_;
    edm::EDGetTokenT<edm::View<reco::CaloMET> > inputCaloMETToken_;
    edm::EDGetTokenT<edm::View<reco::MET> > inputMETToken_;
    
  };
}

//____________________________________________________________________________||
#endif // RecoMET_MuonMET_h
