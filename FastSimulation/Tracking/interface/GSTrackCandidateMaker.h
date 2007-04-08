#ifndef FastSimulation_Tracking_GSTrackCandidateMaker_h
#define FastSimulation_Tracking_GSTrackCandidateMaker_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class TransientInitialStateEstimator;

class GSTrackCandidateMaker : public edm::EDProducer
{
 public:
  
  explicit GSTrackCandidateMaker(const edm::ParameterSet& conf);
  
  virtual ~GSTrackCandidateMaker();
  
  virtual void beginJob (edm::EventSetup const & es);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  edm::ParameterSet conf_;
  
  const MagneticField*  theMagField;
  const TrackerGeometry*  theGeometry;

  double pTMin;
  double maxD0;
  double maxZ0;
  unsigned minRecHits;
    
};

#endif
