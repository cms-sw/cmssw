#ifndef RecoMET_MuonMET_h
#define RecoMET_MuonMET_h
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

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "RecoMET/METAlgorithms/interface/MuonMETAlgo.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"


namespace cms 
{
  // PRODUCER CLASS DEFINITION -------------------------------------
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
    
  };
}

#endif
