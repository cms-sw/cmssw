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
#include "FWCore/Framework/interface/ESWatcher.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"

class MultiTrajectoryStateTransform;

class EgammaHLTPixelMatchElectronAlgo {
public:
  EgammaHLTPixelMatchElectronAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);

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

  std::unique_ptr<MultiTrajectoryStateTransform> mtsTransform_ = nullptr;

  edm::ESHandle<MagneticField> magField_;
  edm::ESHandle<TrackerGeometry> trackerGeom_;

  edm::ESWatcher<IdealMagneticFieldRecord> magneticFieldWatcher_;
  edm::ESWatcher<TrackerDigiGeometryRecord> trackerGeometryWatcher_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
};

#endif  // EgammaHLTPixelMatchElectronAlgo_H
