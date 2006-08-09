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
  double chi2TIB2intstr1, p0TIB2intstr1, err0TIB2intstr1, p1TIB2intstr1, err1TIB2intstr1, p2TIB2intstr1, err2TIB2intstr1;
  double chi2TIB2intstr2, p0TIB2intstr2, err0TIB2intstr2, p1TIB2intstr2, err1TIB2intstr2, p2TIB2intstr2, err2TIB2intstr2;
  double chi2TIB2extstr1, p0TIB2extstr1, err0TIB2extstr1, p1TIB2extstr1, err1TIB2extstr1, p2TIB2extstr1, err2TIB2extstr1;
  double chi2TIB2extstr2, p0TIB2extstr2, err0TIB2extstr2, p1TIB2extstr2, err1TIB2extstr2, p2TIB2extstr2, err2TIB2extstr2;
  double chi2TIB2extstr3, p0TIB2extstr3, err0TIB2extstr3, p1TIB2extstr3, err1TIB2extstr3, p2TIB2extstr3, err2TIB2extstr3;
  double chi2TIB3, p0TIB3, err0TIB3, p1TIB3, err1TIB3, p2TIB3, err2TIB3;
  double chi2TIB3intstr1, p0TIB3intstr1, err0TIB3intstr1, p1TIB3intstr1, err1TIB3intstr1, p2TIB3intstr1, err2TIB3intstr1;
  double chi2TIB3intstr2, p0TIB3intstr2, err0TIB3intstr2, p1TIB3intstr2, err1TIB3intstr2, p2TIB3intstr2, err2TIB3intstr2;
  double chi2TIB3intstr3, p0TIB3intstr3, err0TIB3intstr3, p1TIB3intstr3, err1TIB3intstr3, p2TIB3intstr3, err2TIB3intstr3;
  double chi2TIB3intstr4, p0TIB3intstr4, err0TIB3intstr4, p1TIB3intstr4, err1TIB3intstr4, p2TIB3intstr4, err2TIB3intstr4;
  double chi2TIB3intstr5, p0TIB3intstr5, err0TIB3intstr5, p1TIB3intstr5, err1TIB3intstr5, p2TIB3intstr5, err2TIB3intstr5;
  double chi2TIB3intstr6, p0TIB3intstr6, err0TIB3intstr6, p1TIB3intstr6, err1TIB3intstr6, p2TIB3intstr6, err2TIB3intstr6;
  double chi2TIB3intstr7, p0TIB3intstr7, err0TIB3intstr7, p1TIB3intstr7, err1TIB3intstr7, p2TIB3intstr7, err2TIB3intstr7;
  double chi2TIB3intstr8, p0TIB3intstr8, err0TIB3intstr8, p1TIB3intstr8, err1TIB3intstr8, p2TIB3intstr8, err2TIB3intstr8;
  double chi2TIB3extstr1, p0TIB3extstr1, err0TIB3extstr1, p1TIB3extstr1, err1TIB3extstr1, p2TIB3extstr1, err2TIB3extstr1;
  double chi2TIB3extstr2, p0TIB3extstr2, err0TIB3extstr2, p1TIB3extstr2, err1TIB3extstr2, p2TIB3extstr2, err2TIB3extstr2;
  double chi2TIB3extstr3, p0TIB3extstr3, err0TIB3extstr3, p1TIB3extstr3, err1TIB3extstr3, p2TIB3extstr3, err2TIB3extstr3;
  double chi2TIB3extstr4, p0TIB3extstr4, err0TIB3extstr4, p1TIB3extstr4, err1TIB3extstr4, p2TIB3extstr4, err2TIB3extstr4;
  double chi2TIB3extstr5, p0TIB3extstr5, err0TIB3extstr5, p1TIB3extstr5, err1TIB3extstr5, p2TIB3extstr5, err2TIB3extstr5;
  double chi2TIB3extstr6, p0TIB3extstr6, err0TIB3extstr6, p1TIB3extstr6, err1TIB3extstr6, p2TIB3extstr6, err2TIB3extstr6;
  double chi2TIB3extstr7, p0TIB3extstr7, err0TIB3extstr7, p1TIB3extstr7, err1TIB3extstr7, p2TIB3extstr7, err2TIB3extstr7;
  double chi2TOB1, p0TOB1, err0TOB1, p1TOB1, err1TOB1, p2TOB1, err2TOB1;
  double chi2TOB1rod1, p0TOB1rod1, err0TOB1rod1, p1TOB1rod1, err1TOB1rod1, p2TOB1rod1, err2TOB1rod1;
  double chi2TOB1rod2, p0TOB1rod2, err0TOB1rod2, p1TOB1rod2, err1TOB1rod2, p2TOB1rod2, err2TOB1rod2;
  double chi2TOB2, p0TOB2, err0TOB2, p1TOB2, err1TOB2, p2TOB2, err2TOB2;
  double chi2TOB2rod1, p0TOB2rod1, err0TOB2rod1, p1TOB2rod1, err1TOB2rod1, p2TOB2rod1, err2TOB2rod1;
  double chi2TOB2rod2, p0TOB2rod2, err0TOB2rod2, p1TOB2rod2, err1TOB2rod2, p2TOB2rod2, err2TOB2rod2;
  double chi2TOB, p0TOB, err0TOB, p1TOB, err1TOB, p2TOB, err2TOB;
  double minTIB2, minTIB3, minTOB1, minTOB2, minTOB;
  double minTIB2intstr1, minTIB2intstr2, minTIB2extstr1, minTIB2extstr2, minTIB2extstr3;
  double minTIB3intstr1, minTIB3intstr2, minTIB3intstr3, minTIB3intstr4, minTIB3intstr5, minTIB3intstr6, minTIB3intstr7, minTIB3intstr8;
  double minTIB3extstr1, minTIB3extstr2, minTIB3extstr3, minTIB3extstr4, minTIB3extstr5, minTIB3extstr6, minTIB3extstr7; 
  double minTOB1rod1, minTOB1rod2, minTOB2rod1, minTOB2rod2;
  
  
  TF1* fitfunc;
  TFile* hFile;
  TTree* SiStripLorentzAngleTree;
  TH1F  *hphi, *hnhit;
  TH1F  *htaTIBL2, *hrwTIBL2;
  TH1F  *htaTIBL3, *hrwTIBL3;
  TH1F  *htaTOB1, *hrwTOB1;
  TH1F  *htaTOB2, *hrwTOB2;
  TProfile *hwvst, *hwvsaTIBL2,  *hwvsaTIBL3, *hwvsaTOB, *hwvsaTOBL1, *hwvsaTOBL2;
  TProfile *hwvsaTOBL1rod1, *hwvsaTOBL1rod2, *hwvsaTOBL2rod1, *hwvsaTOBL2rod2;
  TProfile *hwvsaTIBL2intstr1, *hwvsaTIBL2intstr2, *hwvsaTIBL2extstr1, *hwvsaTIBL2extstr2, *hwvsaTIBL2extstr3;
  TProfile *hwvsaTIBL3intstr1, *hwvsaTIBL3intstr2, *hwvsaTIBL3intstr3, *hwvsaTIBL3intstr4, *hwvsaTIBL3intstr5, *hwvsaTIBL3intstr6, *hwvsaTIBL3intstr7, *hwvsaTIBL3intstr8;
  TProfile *hwvsaTIBL3extstr1, *hwvsaTIBL3extstr2, *hwvsaTIBL3extstr3, *hwvsaTIBL3extstr4, *hwvsaTIBL3extstr5, *hwvsaTIBL3extstr6, *hwvsaTIBL3extstr7;
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
