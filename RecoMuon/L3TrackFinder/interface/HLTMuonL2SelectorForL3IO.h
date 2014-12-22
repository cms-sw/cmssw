#ifndef RecoMuon_HLTMuonL2SelectorForL3IO_HLTMuonL2SelectorForL3IO_H
#define RecoMuon_HLTMuonL2SelectorForL3IO_HLTMuonL2SelectorForL3IO_H

//-------------------------------------------------
//
/**  \class HLTMuonL2SelectorForL3IO
 * 
 *   L2 muon selector for L3 IO:
 *   finds L2 muons not previous converted into L3 muons
 *
 *   \author  Benjamin Radburn-Smith - Purdue University
 */
//
//--------------------------------------------------

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
//#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"


namespace edm {class ParameterSet; class Event; class EventSetup;}

//class MuonTrackFinder;
//class MuonServiceProxy;

class HLTMuonL2SelectorForL3IO : public edm::EDProducer {
  public:
  /// constructor with config
  HLTMuonL2SelectorForL3IO(const edm::ParameterSet&);
  
  /// destructor
  virtual ~HLTMuonL2SelectorForL3IO(); 
  
  /// select muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
    
 private:
  // MuonSeed Collection Label
	edm::EDGetTokenT<reco::TrackCollection> l2Src_;
	edm::EDGetTokenT<reco::TrackCollection> l3OISrc_;

};

#endif
