#ifndef FastSimulation_Tracking_TrajectorySeedProducer_h
#define FastSimulation_Tracking_TrajectorySeedProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h" 
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

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
  
  virtual void beginRun(edm::Run const& run, const edm::EventSetup & es) override;
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;
  
  //
  // 1 = PXB, 2 = PXD, 3 = TIB, 4 = TID, 5 = TOB, 6 = TEC, 0 = not valid
  enum SubDet { NotValid, PXB, PXD, TIB, TID, TOB, TEC};
  // 0 = barrel, -1 = neg. endcap, +1 = pos. endcap
  enum Side { BARREL=0, NEG_ENDCAP=-1, POS_ENDCAP=1};
  
  struct LayerSpec {
    std::string name;
    SubDet subDet;
    Side side;
    unsigned int idLayer;
  };
  //
  
  Side setLayerSpecSide(const std::string& layerSpecSide) const;

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
  std::vector< std::vector<LayerSpec> > theLayersInSets;
  //
  
  std::vector<double> originRadius;
  std::vector<double> originHalfLength;
  std::vector<double> originpTMin;

  std::vector<edm::InputTag> primaryVertices;
  std::vector<double> zVertexConstraint;

  bool selectMuons;

  std::vector<const reco::VertexCollection*> vertices;
  double x0, y0, z0;

  // tokens
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken;
  edm::EDGetTokenT<SiTrackerGSMatchedRecHit2DCollection> recHitToken;
  std::vector<edm::EDGetTokenT<reco::VertexCollection> > recoVertexToken;
};

#endif
