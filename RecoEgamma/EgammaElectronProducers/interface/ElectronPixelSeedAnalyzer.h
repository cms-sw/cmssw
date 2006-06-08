#ifndef ElectronPixelSeedAnalyzer_h
#define ElectronPixelSeedAnalyzer_h
  
//
// Package:         RecoEgamma/ElectronTrackSeed
// Class:           ElectronPixelSeedAnalyzer
// 

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronPixelSeedAnalyzer.h,v 1.1 2006/06/02 15:32:44 uberthon Exp $
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
  TH1F *histpt_;
  TH1I *histnrseeds_;
 };
  
#endif
 


