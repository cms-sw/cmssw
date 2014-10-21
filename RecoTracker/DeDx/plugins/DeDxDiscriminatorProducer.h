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

#include <ext/hash_map>


// using namespace edm;
// using namespace reco;
// using namespace std;
// using namespace __gnu_cxx;




class DeDxDiscriminatorProducer : public edm::stream::EDProducer<> {

public:

  explicit DeDxDiscriminatorProducer(const edm::ParameterSet&);
  ~DeDxDiscriminatorProducer();

private:
  virtual void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override;

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


  const TrackerGeometry* m_tracker;

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


  TH3D*        Prob_ChargePath;



   private :
      struct stModInfo{int DetId; int SubDet; float Eta; float R; float Thickness; int NAPV; double Gain;};

      class isEqual{
         public:
                 template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
      };

  __gnu_cxx::hash_map<unsigned int, stModInfo*,  __gnu_cxx::hash<unsigned int>, isEqual > MODsColl;
};

#endif

