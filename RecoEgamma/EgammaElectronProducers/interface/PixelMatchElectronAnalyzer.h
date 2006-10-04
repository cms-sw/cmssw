#ifndef PixelMatchElectronAnalyzer_h
#define PixelMatchElectronAnalyzer_h
  
//
// Package:         RecoEgamma/ElectronTrackSeed
// Class:           PixelMatchElectronAnalyzer
// 

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: PixelMatchElectronAnalyzer.h,v 1.3 2006/07/03 09:45:42 uberthon Exp $
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

class PixelMatchElectronAnalyzer : public edm::EDAnalyzer
{
 public:
  
  explicit PixelMatchElectronAnalyzer(const edm::ParameterSet& conf);
  
  virtual ~PixelMatchElectronAnalyzer();
  
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


  TH1F *histCharge_;
  TH1F *histMass_ ;
  TH1F *histEn_ ;
  TH1F *histEt_ ;
  TH1F *histEta_ ;
  TH1F *histPhi_ ;
  TH1F *histTrPt_ ;
  TH1F *histTrEta_ ;
  TH1F *histTrPhi_ ;

 };
  
#endif
 


