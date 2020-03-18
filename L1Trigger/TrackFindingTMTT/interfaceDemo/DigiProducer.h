#ifndef __DEMONSTRATOR_PRODUCER_DIGIPRODUCER_HPP__
#define __DEMONSTRATOR_PRODUCER_DIGIPRODUCER_HPP__

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerGeometryInfo.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Histos.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"
#include "Demonstrator/DataFormats/interface/DigiKF4Track.hpp"
#include "Demonstrator/DataFormats/interface/DigiHTStub.hpp"
#include "Demonstrator/DataFormats/interface/DigiHTMiniStub.hpp"
#include "Demonstrator/DataFormats/interface/DigiDTCStub.hpp"

#include <vector>
#include <map>
#include <string>

using namespace std;

namespace demo {

/*class Settings;
class Histos;
class TrackFitGeneric;*/

class DigiProducer : public edm::EDProducer {

public:
  explicit DigiProducer(const edm::ParameterSet&);	
  ~DigiProducer(){}

private:

  typedef std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > TTTrackCollection;

  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

private:
  const edm::EDGetTokenT<TrackingParticleCollection> tpInputTag;
  const edm::EDGetTokenT<TMTT::DetSetVec> stubInputTag;
  const edm::EDGetTokenT<TMTT::TTStubAssMap> stubTruthInputTag;
  const edm::EDGetTokenT<TMTT::TTClusterAssMap> clusterTruthInputTag;
  const edm::EDGetTokenT<reco::GenJetCollection> genJetInputTag_;
  
  // Configuration parameters
  TMTT::Settings *settings_;
  vector<string> trackFitters_;
  vector<string> useRZfilter_;
  bool           runRZfilter_;

  TMTT::Histos   *hists_;
  map<string, TMTT::TrackFitGeneric*> fitterWorkerMap_;

  TMTT::TrackerGeometryInfo              trackerGeometryInfo_;
};

}

#endif /* __DEMONSTRATOR_PRODUCER_DIGIPRODUCER_HPP__ */

