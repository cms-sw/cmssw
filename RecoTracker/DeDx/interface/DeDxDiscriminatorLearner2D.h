#ifndef TrackRecoDeDx_DeDxDiscriminatorLearner2D_H
#define TrackRecoDeDx_DeDxDiscriminatorLearner2D_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "RecoTracker/DeDx/interface/DeDxDiscriminatorTools.h"

#include "TFile.h"
#include "TH2F.h"
#include <ext/hash_map>

using namespace edm;
using namespace reco;
using namespace std;
using namespace __gnu_cxx;


class DeDxDiscriminatorLearner2D : public ConditionDBWriter<PhysicsTools::Calibration::HistogramD2D> {

public:

  explicit DeDxDiscriminatorLearner2D(const edm::ParameterSet&);
  ~DeDxDiscriminatorLearner2D();

private:
  virtual void algoBeginJob(const edm::EventSetup&) ;
  virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&);
  virtual void algoEndJob();

  void         Learn(const SiStripRecHit2D* sistripsimplehit, TrajectoryStateOnSurface trajState);

  PhysicsTools::Calibration::HistogramD2D * getNewObject();


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

  string       algoMode;
  string       HistoFile;

  TH2F*        Charge_Vs_Path;

private :
  struct stModInfo{int DetId; int SubDet; float Eta; float R; float Thickness; int NAPV; };

  class isEqual{
  public:
    template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
  };

  hash_map<unsigned int, stModInfo*,  hash<unsigned int>, isEqual > MODsColl;
};

#endif

