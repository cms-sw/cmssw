#ifndef CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleAlgorithm_h
#define CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleAlgorithm_h

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"

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


class SiStripLorentzAngleAlgorithm
{
 public:
  typedef struct {double chi2; int ndf; double p0; double p1; double p2; double errp0; double errp1; double errp2;} histofit ;
  typedef std::map <unsigned int, histofit*> fitmap;
  
  explicit SiStripLorentzAngleAlgorithm(const edm::ParameterSet& conf);
  
  ~SiStripLorentzAngleAlgorithm();
  
  void init(const edm::EventSetup& c);
  
  void fit(fitmap & fitresults); 
  
  void run(const edm::Event& e, const edm::EventSetup& c);
  
  std::string makename(const DetId &detid,bool description=false, bool summary=false);
  
 private:
 
  // const TrackerGeometry::DetIdContainer& Id;
  
  typedef std::map <unsigned int, TProfile*> histomap;
  histomap histos;
  histomap summaryhisto;
  
  //  fitmap fits;
  fitmap summaryfits;
  
  typedef std::vector<std::pair<const TrackingRecHit *,float> > hitanglevector;  
  typedef std::map <const reco::Track *, hitanglevector> trackhitmap;
  
  typedef struct {float thickness; float pitch; LocalVector magfield;} detparameters;
  typedef std::map <unsigned int, detparameters*> detparmap;
  detparmap detmap;
  detparmap summarydetmap;
  
  typedef unsigned short uint16_t;
    
  edm::ParameterSet conf_;
  //  std::string filename_;
  
  //  std::vector<DetId> Detvector;
  
  //  int mtcctibcorr, mtcctobcorr;
  
  // int monodscounter;
  //int monosscounter;
  //int stereocounter;
  int eventcounter, trackcounter, hitcounter, runcounter;
  int runvector[1000];
  
  unsigned int runnr;
  int eventnr;
  int trackcollsize;
  int trajsize;

  TFile* hFile;
  

  const TrackerGeometry * tracker;
  const MagneticField * magfield;

  
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
  
  TDirectory *TIBfwl[4];
  TDirectory *TIBbwl[4];
  
  //TID directories
  
  TDirectory *TIDfwr[4];
  TDirectory *TIDbwr[4];
  
  //TOB directories
  
  TDirectory *TOBfwl[6];
  TDirectory *TOBbwl[6];

  //TEC directories
  
  TDirectory *TECfwr[7];
  TDirectory *TECbwr[7];
};


#endif
