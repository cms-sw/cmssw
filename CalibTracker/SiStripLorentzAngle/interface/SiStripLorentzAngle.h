#ifndef CalibTracker_SiStripLorentzAngle_SiStripLorentzAngle_h
#define CalibTracker_SiStripLorentzAngle_SiStripLorentzAngle_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripLorentzAngle/interface/TrackLocalAngle.h"
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TF1.h>
#include <TProfile.h>

class SiStripLorentzAngle : public edm::EDAnalyzer
{
 public:
  
  explicit SiStripLorentzAngle(const edm::ParameterSet& conf);
  
  virtual ~SiStripLorentzAngle();
  virtual void beginJob(const edm::EventSetup& c);
  virtual void endJob(); 
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  void findtrackangle(const TrajectorySeed& seed,
				       const TrackingRecHitCollection &hits,
				       const edm::Event& e, 
					const edm::EventSetup& es);
  TrajectoryStateOnSurface startingTSOS(const TrajectorySeed& seed)const;
 private:
 
  edm::ParameterSet conf_;
  std::string filename_;
  
  TrackLocalAngle *anglefinder_;
  //  std::vector<PSimHit> theStripHits;
  int run;
  int event;
  int size;
  int module;
  int string;
  int extint;
  int type;
  int layer;
  int eventcounter, eventnumber, trackcounter;
  float angle;
   double chi2TIB2, p0TIB2, err0TIB2, p1TIB2, err1TIB2, p2TIB2, err2TIB2;
  double chi2TIB3, p0TIB3, err0TIB3, p1TIB3, err1TIB3, p2TIB3, err2TIB3;
  double chi2TOB1, p0TOB1, err0TOB1, p1TOB1, err1TOB1, p2TOB1, err2TOB1;
  double chi2TOB2, p0TOB2, err0TOB2, p1TOB2, err1TOB2, p2TOB2, err2TOB2;
  double chi2TOB, p0TOB, err0TOB, p1TOB, err1TOB, p2TOB, err2TOB;
  double minTIB2, minTIB3, minTOB1, minTOB2, minTOB;
  
  TF1* fitfunc;
  
  TF1* fitTIB2;
  TF1* fitTIB3;
  TF1* fitTOB;
  TF1* fitTOB1;
  TF1* fitTOB2;
 
  TFile* hFile;
  TTree* SiStripLorentzAngleTree;
  TH1F  *hphi, *hnhit;
  TH1F  *htaTIBL2, *hrwTIBL2;
  TH1F  *htaTIBL3, *hrwTIBL3;
  TH1F  *htaTOB1, *hrwTOB1;
  TH1F  *htaTOB2, *hrwTOB2;
  TProfile *hwvst, *hwvsaTIBL2,  *hwvsaTIBL3, *hwvsaTOB, *hwvsaTOBL1, *hwvsaTOBL2;
  bool seed_plus;
  PropagatorWithMaterial  *thePropagator;
  PropagatorWithMaterial  *thePropagatorOp;
  KFUpdator *theUpdator;
  Chi2MeasurementEstimator *theEstimator;
  const TransientTrackingRecHitBuilder *RHBuilder;
  const KFTrajectorySmoother * theSmoother;
  const KFTrajectoryFitter * theFitter;
  const TrackerGeometry * tracker;
  const MagneticField * magfield;
  TrajectoryStateTransform tsTransform;
  
};


#endif
