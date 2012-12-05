#ifndef FastSimulation_Tracking_TrajectorySeedProducer_h
#define FastSimulation_Tracking_TrajectorySeedProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <vector>
#include <string>

class TransientInitialStateEstimator;
class MagneticField;
class MagneticFieldMap;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class PTrajectoryStateOnDet;
class ParticlePropagator; 
class PropagatorWithMaterial;

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TrajectorySeedProducer : public edm::EDProducer
{
 public:
  
  explicit TrajectorySeedProducer(const edm::ParameterSet& conf);
  
  virtual ~TrajectorySeedProducer();
  
  virtual void beginRun(edm::Run & run, const edm::EventSetup & es);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  /// A mere copy (without memory leak) of an existing tracking method
  void stateOnDet(const TrajectoryStateOnSurface& ts,
		  unsigned int detid,
		  PTrajectoryStateOnDet& pts) const;
  
  /// Check that the seed is compatible with a track coming from within
  /// a cylinder of radius originRadius, with a decent pT.
  bool compatibleWithBeamAxis(GlobalPoint& gpos1, 
			      GlobalPoint& gpos2,
			      double error,
			      bool forward,
			      unsigned algo) const;

 private:

  const MagneticField*  theMagField;
  const MagneticFieldMap*  theFieldMap;
  const TrackerGeometry*  theGeometry;
  PropagatorWithMaterial* thePropagator;

  std::vector<double> pTMin;
  std::vector<double> maxD0;
  std::vector<double> maxZ0;
  std::vector<unsigned> minRecHits;
  edm::InputTag hitProducer;
  edm::InputTag theBeamSpot;

  bool seedCleaning;
  bool rejectOverlaps;
  unsigned int absMinRecHits;
  std::vector<std::string> seedingAlgo;
  std::vector<unsigned int> numberOfHits;
  ///// TO BE REMOVED (AG)
  std::vector<unsigned int> firstHitSubDetectorNumber;
  std::vector<unsigned int> secondHitSubDetectorNumber;
  std::vector<unsigned int> thirdHitSubDetectorNumber;
  std::vector< std::vector<unsigned int> > firstHitSubDetectors;
  std::vector< std::vector<unsigned int> > secondHitSubDetectors;
  std::vector< std::vector<unsigned int> > thirdHitSubDetectors;
  /////
  bool newSyntax;
  std::vector<std::string> layerList;

  std::vector<double> originRadius;
  std::vector<double> originHalfLength;
  std::vector<double> originpTMin;

  std::vector<edm::InputTag> primaryVertices;
  std::vector<double> zVertexConstraint;

  bool selectMuons;

  std::vector<const reco::VertexCollection*> vertices;
  double x0, y0, z0;

};

#endif
