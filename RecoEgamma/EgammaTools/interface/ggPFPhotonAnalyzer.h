#ifndef ggPFPhotonAnalyzer_h
#define ggPFPhotonAnalyzer_h
// system include files
#include <memory>
#include <string>
#include <iostream>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "RecoEgamma/EgammaTools/interface/ggPFPhotons.h"

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>

class ggPFPhotonAnalyzer : public edm::EDAnalyzer {
 public:
  
  explicit ggPFPhotonAnalyzer(const edm::ParameterSet& iConfig);
  
  ~ggPFPhotonAnalyzer();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup& es);
  
  virtual void beginRun(const edm::Run & r, const edm::EventSetup & c);
  virtual void endJob();
  
 private:
  edm::InputTag   PFPhotonTag_;
  edm::InputTag   recoPhotonTag_;
  edm::InputTag ebReducedRecHitCollection_;
  edm::InputTag eeReducedRecHitCollection_;
  edm::InputTag esRecHitCollection_;
  edm::InputTag beamSpotCollection_;
  const CaloSubdetectorGeometry* geomBar_;
  const CaloSubdetectorGeometry* geomEnd_;
  TFile *tf1;
  TTree* pf;
  TTree* pfclus;
  int isConv_;
  int hasSLConv_;
  float PFPS1_;
  float PFPS2_;
  float MustE_;
  float MustEOut_;
  float PFLowCE_;
  float PFdEta_;
  float PFdPhi_;
  float PFClusRMS_;
  float PFClusRMSMust_;
  float VtxZ_;
  float VtxZErr_;
};
#endif
