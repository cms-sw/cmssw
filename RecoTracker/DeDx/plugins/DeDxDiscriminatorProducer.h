#ifndef TrackRecoDeDx_DeDxDiscriminatorProducer_H
#define TrackRecoDeDx_DeDxDiscriminatorProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h" 

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

#include "RecoTracker/DeDx/interface/DeDxDiscriminatorTools.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"


#include "TH3F.h"
#include "TChain.h"

#include <unordered_map>


namespace DeDxDiscriminatorProducerDetails {
   struct stModInfo{int DetId; int SubDet; float Eta; float R; float Thickness; int NAPV; double Gain;};
   struct Cache {
       const TrackerGeometry* m_tracker;
       PhysicsTools::Calibration::HistogramD3D DeDxMap_;
       mutable std::unordered_map<unsigned int, stModInfo> MODsColl;

  };

}


class DeDxDiscriminatorProducer : public edm::stream::EDProducer<edm::RunCache<DeDxDiscriminatorProducerDetails::Cache>> {

public:

  explicit DeDxDiscriminatorProducer(const edm::ParameterSet&);
  ~DeDxDiscriminatorProducer();

public:
  using Cache = DeDxDiscriminatorProducerDetails::Cache;

  static std::shared_ptr<Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&, GlobalCache const*);
  static void globalEndRun(edm::Run const&, edm::EventSetup const&, RunContext const*){}

  virtual void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  double GetProbability(const SiStripCluster*   cluster, TrajectoryStateOnSurface trajState,const uint32_t &);
  double ComputeDiscriminator (std::vector<double>& vect_probs);
  int    ClusterSaturatingStrip(const SiStripCluster*   cluster,const uint32_t &);
  void   MakeCalibrationMap();



  // ----------member data ---------------------------
  edm::EDGetTokenT<TrajTrackAssociationCollection>   m_trajTrackAssociationTag;
  edm::EDGetTokenT<reco::TrackCollection>  m_tracksTag;

  bool usePixel;
  bool useStrip;
  double MeVperADCPixel;
  double MeVperADCStrip;

  std::string                       m_calibrationPath;
  bool                              useCalibration;
  bool				    shapetest;


  PhysicsTools::Calibration::HistogramD3D DeDxMap_;

  double       MinTrackMomentum;
  double       MaxTrackMomentum;
  double       MinTrackEta;
  double       MaxTrackEta;
  unsigned int MaxNrStrips;
  unsigned int MinTrackHits;
  double       MaxTrackChiOverNdf;

  unsigned int Formula;
  std::string       Reccord;
  std::string       ProbabilityMode;


  std::unique_ptr<TH3F>        Prob_ChargePath;

};

#endif

