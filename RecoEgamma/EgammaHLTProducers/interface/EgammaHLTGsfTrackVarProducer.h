#ifndef RECOEGAMMA_EGAMMAHLTPRODUCERS_EGAMMAHLTGSFTRACKVARPRODUCER
#define RECOEGAMMA_EGAMMAHLTPRODUCERS_EGAMMAHLTGSFTRACKVARPRODUCER


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//this class is designed to calculate dEtaIn,dPhiIn gsf track - supercluster pairs
//it can take as input std::vector<Electron> which the gsf track-sc is already done
//or it can run over the std::vector<GsfTrack> directly in which case it will pick the smallest dEta,dPhi
//the dEta, dPhi do not have to be from the same track
//it can optionally set dEta, dPhi to 0 based on the number of tracks found

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTGsfTrackVarProducer : public edm::stream::EDProducer<> {
 private:
  class TrackExtrapolator {
    unsigned long long cacheIDTDGeom_;
    unsigned long long cacheIDMagField_;
    
    edm::ESHandle<MagneticField> magField_;
    edm::ESHandle<TrackerGeometry> trackerHandle_;
    
    MultiTrajectoryStateMode mtsMode_; 
    const MultiTrajectoryStateTransform * mtsTransform_; //we own it
    
  public:
  TrackExtrapolator():cacheIDTDGeom_(0),cacheIDMagField_(0),mtsTransform_(0){}
    TrackExtrapolator(const TrackExtrapolator& rhs);
    ~TrackExtrapolator(){delete mtsTransform_;}
    TrackExtrapolator* operator=(const TrackExtrapolator& rhs);
      
    void setup(const edm::EventSetup& iSetup);
    
    GlobalPoint extrapolateTrackPosToPoint(const reco::GsfTrack& gsfTrack,const GlobalPoint& pointToExtrapTo);
    GlobalVector extrapolateTrackMomToPoint(const reco::GsfTrack& gsfTrack,const GlobalPoint& pointToExtrapTo);
    
    edm::ESHandle<TrackerGeometry> trackerGeomHandle()const{return trackerHandle_;}
    const MultiTrajectoryStateTransform * mtsTransform()const{return mtsTransform_;}
    const MultiTrajectoryStateMode* mtsMode()const{return &mtsMode_;}
  };
  
 public:
  explicit EgammaHLTGsfTrackVarProducer(const edm::ParameterSet&);
  ~EgammaHLTGsfTrackVarProducer();
  void produce(edm::Event&, const edm::EventSetup&) override; 
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandTag_;
  const edm::EDGetTokenT<reco::ElectronCollection> inputCollectionTag1_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> inputCollectionTag2_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;
  
  TrackExtrapolator trackExtrapolator_;
  const int upperTrackNrToRemoveCut_;
  const int lowerTrackNrToRemoveCut_;
};

#endif
