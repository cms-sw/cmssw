#ifndef RecoMuon_GlobalTrackingTools_ MuonErrorMatrixAnalyzer_H
#define RecoMuon_GlobalTrackingTools_ MuonErrorMatrixAnalyzer_H

// -*- C++ -*-
//
// Package:    MuonErrorMatrixAnalyzer
// Class:      MuonErrorMatrixAnalyzer
// 
/**\class MuonErrorMatrixAnalyzer

 Description: edproducer, duplicating a collection of track, adjusting their error matrix

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri May  4 15:15:44 CDT 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DataFormats/TrackReco/interface/Track.h>
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>
#include <MagneticField/Engine/interface/MagneticField.h>

//#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>
#include <TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryParameters.h>

#include <TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h>
#include "RecoMuon/GlobalTrackingTools/interface/ErrorMatrix.h"
#include <DataFormats/GeometrySurface/interface/Plane.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h>

#include "TH1F.h"
#include "TH2F.h"

//For phi angle
#include <cmath>
#include "DataFormats/Math/interface/deltaPhi.h" 

//add these for associator

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"


//
// class decleration
//

class MuonErrorMatrixAnalyzer : public edm::EDAnalyzer {
 public:
  explicit MuonErrorMatrixAnalyzer(const edm::ParameterSet&);
  ~MuonErrorMatrixAnalyzer();
  
  
 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  void analyze_from_errormatrix(const edm::Event&, const edm::EventSetup&);
  void analyze_from_pull(const edm::Event&, const edm::EventSetup&);


  double term(const AlgebraicSymMatrix55 & curv, int i, int j);
  
  // ----------member data ---------------------------
  std::string theCategory;

  edm::InputTag theTrackLabel;
  edm::InputTag trackingParticleLabel;

  //name of the associator
  std::string theAssocLabel;
  edm::ESHandle<TrackAssociatorBase> theAssociator;

  ErrorMatrix * theErrorMatrixStore;
  edm::ParameterSet theErrorMatrixStore_Error_pset;

  ErrorMatrix * theErrorMatrixStore_Spread;
  edm::ParameterSet theErrorMatrixStore_Spread_pset;

  TFile * thePlotFile;
  std::string thePlotFileName;

  edm::ESHandle<MagneticField> theField;

  typedef TH1* TH1ptr;
  TH1ptr* theHist_array[15];
  
  inline uint index(TProfile3D * pf, uint i ,uint j,uint k)   {return (((i*pf->GetNbinsY())+j) * pf->GetNbinsZ())+k;}
  uint maxIndex(TProfile3D * pf)  {return pf->GetNbinsX()*pf->GetNbinsY()*pf->GetNbinsZ();}

  struct extractRes{
    double corr;
    double x;
    double y;
  };
  extractRes extract(TH2 * h2);
};
#endif
