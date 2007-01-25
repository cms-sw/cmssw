#ifndef CalibTracker_SiStripLorentzAngle_SiStripLorentzAngle_h
#define CalibTracker_SiStripLorentzAngle_SiStripLorentzAngle_h

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH1.h>
#include <TF1.h>
#include <TProfile.h>
#include <TFolder.h>
#include <TDirectory.h>
#include <TAxis.h>
#include <TMath.h>


class SiStripLorentzAngle : public edm::EDAnalyzer
{
 public:
  
  explicit SiStripLorentzAngle(const edm::ParameterSet& conf);
  
  virtual ~SiStripLorentzAngle();
  
  virtual void beginJob(const edm::EventSetup& c);
  
  virtual void endJob(); 
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    
  const char* makename(DetId detid);
  
  const char* makedescription(DetId detid);
  
 private:
 
  const TrackerGeometry::DetIdContainer& Id;
  
  typedef std::map <int, TProfile*> histomap;
  histomap histos;
  
  typedef struct {double chi2; int ndf; double p0; double p1; double p2; double errp0; double errp1; double errp2;} histofit ;
  typedef std::map <int, histofit*> fitmap;
  fitmap fits;
  
  typedef std::vector<std::pair<const TrackingRecHit *,float> > hitanglevector;  
  typedef std::map <const reco::Track *, hitanglevector> trackhitmap;
  
  typedef struct {float thickness; float pitch;} detparameters;
  typedef std::map <int, detparameters*> detparmap;
  detparmap detmap;
  
  typedef unsigned short uint16_t;
    
  edm::ParameterSet conf_;
  std::string filename_;
  
  std::vector<DetId> Detvector;
  
  int mtcctibcorr, mtcctobcorr;
  
  int monodscounter;
  int monosscounter;
  int stereocounter;
  int eventcounter, trackcounter, hitcounter, runcounter;
  int runvector[1000];
  
  int run;
  int event;
  int size;
  int module;
  int string;
  int rod;
  int extint;
  int bwfw;
  int wheel;
  int type;
  int layer, TIBlayer2, TIBlayer3, TOBlayer1, TOBlayer5, TECwheel1;
  int ParticleCharge;
  float TrackLocalAngle;
  float trackproj;
  float signprojcorrection;
  float localmagfield;
  float tangent;
  int monostereo;
  float momentum, pt, chi2, chi2norm;
  float ndof;
  int hitscharge;
  float ThetaTrack;
  float PhiTrack;
  int SeedType;
  int SeedSize;
  int SeedLayer;
  int TOB_YtoGlobalSign;
  
  int hitspertrack;
  int trackcollsize;
  int trajsize;
  int MONOhitsTIBL2collection;
  int STEREOhitsTIBL2collection;
  int hitsTIBL3collection;
  int hitsTOBL1collection;
  int hitsTOBL5collection;
  int hitsTOBcoll;
  int MONOhitsTECcollection;
  int STEREOhitsTECcollection;
  int hitcharge;
  int HitSize;
  
  int MONOhitschargeTIBL2[1000];
  int STEREOhitschargeTIBL2[1000];
  int hitschargeTIBL3[1000];
  int hitschargeTOBL1[1000];
  int hitschargeTOBL5[1000];
  int MONOhitschargeTEC[1000];
  int STEREOhitschargeTEC[1000];  
  
  int TOB_YtoGlobalSignColl[1000];
 
  LocalVector localmagdir;
      
  TF1* fitfunc;
  TFile* hFile;
  TTree* SiStripLorentzAngleTree;
  TTree* TrackHitTree;
  
  TH1F *CollHitSizeTOBL1, *CollHitSizeTOBL5, *CollHitSizeTIBL2mono, *CollHitSizeTIBL2stereo, *CollHitSizeTIBL3;
  TH1F *TrackHitSizeTOBL1, *TrackHitSizeTOBL5, *TrackHitSizeTIBL2mono, *TrackHitSizeTIBL2stereo, *TrackHitSizeTIBL3;
  
  const TransientTrackingRecHitBuilder *RHBuilder;
  const TrackerGeometry * tracker;
  const MagneticField * magfield;
  TrajectoryStateTransform tsTransform;
    
  //Directory hierarchy  
  
  TDirectory *histograms;
  TDirectory *summary; 
  TDirectory *sizesummary; 
  
  //TIB-TID-TOB-TEC    
  
  TDirectory *TIB;
  TDirectory *TOB;
  TDirectory *TID;
  TDirectory *TEC;
  
  //Forward-Backward
  
  TDirectory *TIBfw;
  TDirectory *TIDfw;
  TDirectory *TOBfw;
  TDirectory *TECfw;
  
  TDirectory *TIBbw;
  TDirectory *TIDbw;
  TDirectory *TOBbw;
  TDirectory *TECbw; 
  
  //TIB directories
  
  TDirectory *TIBfw1;
  TDirectory *TIBfw2;
  TDirectory *TIBfw3;
  TDirectory *TIBfw4;
  
  TDirectory *TIBbw1;
  TDirectory *TIBbw2;
  TDirectory *TIBbw3;
  TDirectory *TIBbw4;
  
  //TID directories
  
  TDirectory *TIDfw1;
  TDirectory *TIDfw2;
  TDirectory *TIDfw3;
  
  TDirectory *TIDbw1;
  TDirectory *TIDbw2;
  TDirectory *TIDbw3; 
  
  //TOB directories
  
  TDirectory *TOBfw1;
  TDirectory *TOBfw2;
  TDirectory *TOBfw3;
  TDirectory *TOBfw4;
  TDirectory *TOBfw5;
  TDirectory *TOBfw6;
  
  TDirectory *TOBbw1;
  TDirectory *TOBbw2;
  TDirectory *TOBbw3;
  TDirectory *TOBbw4;
  TDirectory *TOBbw5;
  TDirectory *TOBbw6;
  
  //TEC directories
  
  TDirectory *TECfw1;
  TDirectory *TECfw2;
  TDirectory *TECfw3;
  TDirectory *TECfw4;
  TDirectory *TECfw5;
  TDirectory *TECfw6;
  TDirectory *TECfw7;
  TDirectory *TECfw8;
  TDirectory *TECfw9;
  
  TDirectory *TECbw1;
  TDirectory *TECbw2;
  TDirectory *TECbw3;
  TDirectory *TECbw4;
  TDirectory *TECbw5;
  TDirectory *TECbw6;
  TDirectory *TECbw7;
  TDirectory *TECbw8;
  TDirectory *TECbw9;
  
};


#endif
