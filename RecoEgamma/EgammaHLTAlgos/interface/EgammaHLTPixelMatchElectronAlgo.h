#ifndef EgammaHLTPixelMatchElectronAlgo_H
#define EgammaHLTPixelMatchElectronAlgo_H

/** \class EgammaHLTPixelMatchElectronAlgo
 
 * Class to reconstruct electron tracks from electron pixel seeds
 *  keep track of information about the initiating supercluster
 *
 * \author Monica Vazquez Acosta (CERN)
 *
 ************************************************************/

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class MultiTrajectoryStateTransform;

class EgammaHLTPixelMatchElectronAlgo {
public:
  EgammaHLTPixelMatchElectronAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);

  ~EgammaHLTPixelMatchElectronAlgo();

  //disabling the ability to copy this module (lets hope nobody was actually copying it before as Bad Things (TM) would have happened)
private:
  EgammaHLTPixelMatchElectronAlgo(const EgammaHLTPixelMatchElectronAlgo& rhs) {}
  EgammaHLTPixelMatchElectronAlgo& operator=(const EgammaHLTPixelMatchElectronAlgo& rhs) { return *this; }

public:
  void setupES(const edm::EventSetup& setup);
  void run(edm::Event&, reco::ElectronCollection&);

private:
  // create electrons from tracks
  //void process(edm::Handle<reco::TrackCollection> tracksH, reco::ElectronCollection & outEle, Global3DPoint & bs);
  void process(edm::Handle<reco::TrackCollection> tracksH,
               edm::Handle<reco::GsfTrackCollection> gsfTracksH,
               reco::ElectronCollection& outEle,
               Global3DPoint& bs);
  bool isInnerMostWithLostHits(const reco::GsfTrackRef&, const reco::GsfTrackRef&, bool&);

  edm::EDGetTokenT<reco::TrackCollection> trackProducer_;
  edm::EDGetTokenT<reco::GsfTrackCollection> gsfTrackProducer_;
  bool useGsfTracks_;
  edm::EDGetTokenT<reco::BeamSpot> bsProducer_;

  MultiTrajectoryStateTransform* mtsTransform_;

  edm::ESHandle<MagneticField> magField_;
  edm::ESHandle<TrackerGeometry> trackerGeom_;
  int unsigned long long cacheIDTDGeom_;
  unsigned long long cacheIDMagField_;
};

#endif  // EgammaHLTPixelMatchElectronAlgo_H
