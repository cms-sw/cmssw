#ifndef ElectronPixelSeedAnalyzer_h
#define ElectronPixelSeedAnalyzer_h
  
//
// Package:         RecoEgamma/ElectronTrackSeed
// Class:           ElectronPixelSeedAnalyzer
// 

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronPixelSeedAnalyzer.h,v 1.2 2006/06/08 16:54:41 uberthon Exp $
//
//
  
  
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "DataFormats/Common/interface/EDProduct.h"
 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class MagneticField;  
class TFile;
class TH1F;
class TH1I;
class TTree;

class ElectronPixelSeedAnalyzer : public edm::EDAnalyzer
{
 public:
  
  explicit ElectronPixelSeedAnalyzer(const edm::ParameterSet& conf);
  
  virtual ~ElectronPixelSeedAnalyzer();
  
  virtual void beginJob(edm::EventSetup const& iSetup);
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  
 private:

  TrajectoryStateTransform transformer_;
  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<MagneticField> theMagField;
  TFile *histfile_;
  TTree *tree_;
  float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10];
  float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10];
  float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10];
  TH1F *histeMC_;
  TH1F *histp_;
  TH1F *histeclu_;
  TH1F *histpt_;
  TH1F *histptMC_;
  TH1F *histetclu_;
  TH1F *histeffpt_;
  TH1F *histeta_;
  TH1F *histetaMC_;
  TH1F *histetaclu_;
  TH1F *histeffeta_;
  TH1F *histq_;
  TH1F *histeoverp_;
  TH1I *histnrseeds_;
  TH1I *histnbseeds_;
  TH1I *histnbclus_;
 };
  
#endif
 


