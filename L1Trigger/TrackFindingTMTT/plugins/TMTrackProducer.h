#ifndef __TMTRACKPRODUCER_H__
#define __TMTRACKPRODUCER_H__

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

#include <vector>
#include <map>
#include <string>

using namespace std;

namespace TMTT {

class Settings;
class Histos;
class TrackFitGeneric;

class TMTrackProducer : public edm::EDProducer {

public:
  explicit TMTrackProducer(const edm::ParameterSet&);	
  ~TMTrackProducer(){}

private:

  typedef std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > TTTrackCollection;

  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

private:
  edm::EDGetTokenT<DetSetVec> stubInputTag;
  edm::EDGetTokenT<TrackingParticleCollection> tpInputTag;
  edm::EDGetTokenT<TTStubAssMap> stubTruthInputTag;
  edm::EDGetTokenT<TTClusterAssMap> clusterTruthInputTag;
  edm::EDGetTokenT<reco::GenJetCollection> genJetInputTag_;

  // Configuration parameters
  Settings *settings_;
  vector<string> trackFitters_;
  vector<string> useRZfilter_;
  bool           runRZfilter_;

  Histos   *hists_;
  map<string, TrackFitGeneric*> fitterWorkerMap_;

  TrackerGeometryInfo              trackerGeometryInfo_;
};

}

#endif

