#ifndef TrackRecoDeDx_DeDxDiscriminatorLearner_H
#define TrackRecoDeDx_DeDxDiscriminatorLearner_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "RecoTracker/DeDx/interface/DeDxDiscriminatorTools.h"

#include "TFile.h"
#include "TH3F.h"
#include <ext/hash_map>

//using namespace edm;
//using namespace reco;
//using namespace std;
//using namespace __gnu_cxx;


class DeDxDiscriminatorLearner : public ConditionDBWriter<PhysicsTools::Calibration::HistogramD3D> {

public:

  explicit DeDxDiscriminatorLearner(const edm::ParameterSet&);
  ~DeDxDiscriminatorLearner();

private:
  virtual void algoBeginJob(const edm::EventSetup&) ;
  virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&);
  virtual void algoEndJob();

  void         Learn(const SiStripCluster*   cluster, TrajectoryStateOnSurface trajState);

  PhysicsTools::Calibration::HistogramD3D * getNewObject();


  // ----------member data ---------------------------
  edm::InputTag                     m_trajTrackAssociationTag;
  edm::InputTag                     m_tracksTag;

  bool   usePixel;
  bool   useStrip;
  double MeVperADCPixel;
  double MeVperADCStrip;

  const TrackerGeometry* m_tracker;

  double       MinTrackMomentum;
  double       MaxTrackMomentum;
  double       MinTrackEta;
  double       MaxTrackEta;
  unsigned int MaxNrStrips;
  unsigned int MinTrackHits;
  double       MaxTrackChiOverNdf;


  double P_Min;
  double P_Max;
  int    P_NBins; 
  double Path_Min;
  double Path_Max;
  int    Path_NBins;
  double Charge_Min;
  double Charge_Max;
  int    Charge_NBins;


  std::string       algoMode;
  std::string       HistoFile;

  TH3F*        Charge_Vs_Path;

private :
  struct stModInfo{int DetId; int SubDet; float Eta; float R; float Thickness; int NAPV; };

  class isEqual{
  public:
    template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
  };

  __gnu_cxx::hash_map<unsigned int, stModInfo*,  __gnu_cxx::hash<unsigned int>, isEqual > MODsColl;
};

#endif

