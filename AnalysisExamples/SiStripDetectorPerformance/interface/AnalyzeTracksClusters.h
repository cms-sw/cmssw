// created by Livio Fano'
#ifndef AnalysisExamples_SiStripDetectorPerformance_AnalyzeTracksClusters_h
#define AnalysisExamples_SiStripDetectorPerformance_AnalyzeTracksClusters_h
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackLocalAngle.h"
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
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TProfile.h>

class TTree;


class AnalyzeTracksClusters : public edm::EDAnalyzer
{
 public:
  
  explicit AnalyzeTracksClusters(const edm::ParameterSet& conf);
  
  virtual ~AnalyzeTracksClusters();

  virtual void beginJob(const edm::EventSetup& c);
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  void FillHisto(unsigned int, double, double, double, double);
  std::vector<short> toycluster(int);
  std::vector<short> smear(std::vector<short>);
 private:
  edm::ParameterSet conf_;
  // DeDx* myDeDx;
  TrackLocalAngle *anglefinder_;
  const TrackerGeometry * tracker;
  const MagneticField * magfield;

  int itk;
  int nev;
  // tree (track properties):
  double mean;
  double mean01;
  double mean02;
  double mean03;
  double mean05;
  double median;
  double harm1;
  double harm2;
  double harm12;
  double harm13;
  double p;
  double pt;
  double eta;
  double phi;
  double dp;
  double dp2;
  double dr;
  int n;
  int pid;
  

  // histos (rechit properties):
  TFile* hFile;
  TTree * nt;


  int DTtrig;
  int NoDTtrig;
  int DTOnlytrig;
  int CSCtrig;
  int othertrig;

  int Ntk;
  float p_tk[100];
  float pt_tk[100];
  float eta_tk[100];
  float phi_tk[100];
  int nhits_tk[100];
  int TrigBits[6];
  
  int Subid[10000000];
  int Layer[10000000];
  float Clu_ch[10000000];
  float Clu_ang[10000000];
  float Clu_size[10000000];
  float Clu_bar[10000000];
  int Clu_1strip[10000000];
  int Clu_rawid[10000000];
  int Nclu;
  int Nclu_matched;

  int Subid_all[10000000];
  int Layer_all[10000000];
  float Clu_ch_all[10000000];
  float Clu_ang_all[10000000];
  float Clu_size_all[10000000];
  float Clu_bar_all[10000000];
  int Clu_1strip_all[10000000];
  int Clu_rawid_all[10000000];

  int Nclu_all;
  int Nclu_all_matched;
  int Nclu_st;
  int Nclu_rphi;
    

};


#endif
