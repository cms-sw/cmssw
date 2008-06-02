#ifndef Type1MET_MuonMET_h
#define Type1MET_MuonMET_h
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

#include "JetMETCorrections/Type1MET/interface/MuonMETAlgo.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

//using namespace std;

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
    std::string metType;
    edm::InputTag inputUncorMetLabel;
    edm::InputTag inputMuonsLabel;
    double muonPtMin;
    double muonEtaRange;
    double muonTrackD0Max;
    double muonTrackDzMax;
    int    muonNHitsMin;
    double muonDPtMax;
    double muonChiSqMax;
    bool   muonDepositCor;

    TrackDetectorAssociator   trackAssociator_;
    TrackAssociatorParameters trackAssociatorParameters_;
  };
}

#endif
