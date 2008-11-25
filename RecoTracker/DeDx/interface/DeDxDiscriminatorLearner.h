#ifndef TrackRecoDeDx_DeDxDiscriminatorLearner_H
#define TrackRecoDeDx_DeDxDiscriminatorLearner_H
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h" 

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TROOT.h"

#include <ext/hash_map>


using namespace edm;
using namespace reco;
using namespace std;
using namespace __gnu_cxx;



//
// class declaration
//

class DeDxDiscriminatorLearner : public ConditionDBWriter<PhysicsTools::Calibration::HistogramD2D> {

public:

  explicit DeDxDiscriminatorLearner(const edm::ParameterSet&);
  ~DeDxDiscriminatorLearner();

private:
  virtual void algoBeginJob(const edm::EventSetup&) ;
  virtual void algoAnalyze(edm::Event&, const edm::EventSetup&);
  virtual void algoEndJob();

  PhysicsTools::Calibration::HistogramD2D * getNewObject();


  bool IsFarFromBorder(TrajectoryStateOnSurface trajState, const uint32_t detid, const edm::EventSetup* iSetup);
  
  double ComputeChargeOverPath(const SiStripRecHit2D* sistripsimplehit,TrajectoryStateOnSurface trajState, const edm::EventSetup* iSetup,  const Track* track, double trajChi2OverN);

  // ----------member data ---------------------------
  edm::InputTag                     m_trajTrackAssociationTag;
  edm::InputTag                     m_tracksTag;

  bool usePixel;
  bool useStrip;
  double MeVperADCPixel;
  double MeVperADCStrip;

  // const edm::EventSetup* iSetup_;
  //  const edm::Event*      iEvent_;
  //  const TrackerGeometry* m_tracker;

  TFile*       MapFile;
  std::string  MapFileName;

  double       MinTrackMomentum;
  double       MaxTrackMomentum;
  double       MinTrackEta;
  double       MaxTrackEta;
  unsigned int MaxNrStrips;
  unsigned int MinTrackHits;
  double       MaxTrackChiOverNdf;
  bool         AllowSaturation;

  bool         DiscriminatorMode;
  unsigned int Formula;

  InputTag     TrackProducer;
  InputTag     TrajToTrackProducer;

  TH2F*        Charge_Vs_Path_Barrel;
  TH2F*        Charge_Vs_Path_Endcap;

  TH2F*        PCharge_Vs_Path_Barrel;
  TH2F*        PCharge_Vs_Path_Endcap;

  vector<float> MeasurementProbabilities;

private :
  struct stModInfo{int DetId; int SubDet; float Eta; float R; float Thickness; int NAPV; };

  class isEqual{
  public:
    template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
  };

  hash_map<unsigned int, stModInfo*,  hash<unsigned int>, isEqual > MODsColl;
};

#endif

