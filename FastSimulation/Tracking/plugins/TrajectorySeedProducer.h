#ifndef FastSimulation_Tracking_TrajectorySeedProducer_h
#define FastSimulation_Tracking_TrajectorySeedProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <vector>
#include <string>

class TransientInitialStateEstimator;
class MagneticField;
class MagneticFieldMap;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class PTrajectoryStateOnDet;
class ParticlePropagator; 

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
  
  virtual void beginJob (edm::EventSetup const & es);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  void stateOnDet(const TrajectoryStateOnSurface& ts,
		  unsigned int detid,
		  PTrajectoryStateOnDet& pts) const;
  
  bool compatibleWithVertex(GlobalPoint& gpos1, 
			    GlobalPoint& gpos2,
			    unsigned algo) const; 

 private:

  const MagneticField*  theMagField;
  const MagneticFieldMap*  theFieldMap;
  const TrackerGeometry*  theGeometry;

  std::vector<double> pTMin;
  std::vector<double> maxD0;
  std::vector<double> maxZ0;
  std::vector<unsigned> minRecHits;
  edm::InputTag hitProducer;


  bool seedCleaning;
  bool rejectOverlaps;
  std::vector<std::string> seedingAlgo;
  std::vector<unsigned int> numberOfHits;
  std::vector<unsigned int> firstHitSubDetectorNumber;
  std::vector<unsigned int> secondHitSubDetectorNumber;
  std::vector<unsigned int> thirdHitSubDetectorNumber;
  std::vector< std::vector<unsigned int> > firstHitSubDetectors;
  std::vector< std::vector<unsigned int> > secondHitSubDetectors;
  std::vector< std::vector<unsigned int> > thirdHitSubDetectors;

  std::vector<double> originRadius;
  std::vector<double> originHalfLength;
  std::vector<double> originpTMin;

};

#endif
