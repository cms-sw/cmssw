
// -*- C++ -*-
//
// Package:    METAnalyzer
// Class:      METAnalyzer
//
/**\class METAnalyzer METAnalyzer.cc MitHig/METAnalyzer/src/METAnalyzer.cc

   Description: <one line class summary>

   Implementation:
   Prepare the Treack Tree for analysis
*/
//
// Original Author:  Yen-Jie Lee
//         Created:  Wed 2011/10/19 14:27:00 CEST 2011
// $Id: METAnalyzer.cc,v 1.4 2011/11/10 10:20:21 yjlee Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <map>

// CMSSW user include files
#include "DataFormats/Common/interface/DetSetAlgorithm.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

// Root include files
#include "TTree.h"

using namespace std;
using namespace edm;
using namespace reco;

//
// class decleration
//

#define PI 3.14159265358979

#define MAXMETS 50
#define MAXVTX 100

struct METEvent{

  // event information
  int nRun;
  int nEv;
  int nLumi;
  int nBX;

  int nMET;
  float METEt[MAXMETS];
  float METPhi[MAXMETS];
  float METSumEt[MAXMETS];

};

class METAnalyzer : public edm::EDAnalyzer {
public:
  explicit METAnalyzer(const edm::ParameterSet&);
  ~METAnalyzer();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void fillMETs(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  // ----------member data ---------------------------

  bool doMET_;

  std::string qualityString_;

  edm::Service<TFileService> fs;
  edm::InputTag METSrc_;

  // Root object
  TTree* metTree_;

  METEvent pev_;

};

//--------------------------------------------------------------------------------------------------
METAnalyzer::METAnalyzer(const edm::ParameterSet& iConfig)

{
  doMET_             = iConfig.getUntrackedParameter<bool>  ("doMET",true);
  METSrc_ = iConfig.getParameter<edm::InputTag>("METSrc");
}

//--------------------------------------------------------------------------------------------------
METAnalyzer::~METAnalyzer()
{
}

//--------------------------------------------------------------------------------------------------
void
METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  pev_.nEv = (int)iEvent.id().event();
  pev_.nRun = (int)iEvent.id().run();
  pev_.nLumi = (int)iEvent.luminosityBlock();
  pev_.nBX = (int)iEvent.bunchCrossing();

  if (doMET_) fillMETs(iEvent, iSetup);
  metTree_->Fill();
}

//--------------------------------------------------------------------------------------------------
void
METAnalyzer::fillMETs(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  Handle<edm::View<reco::MET> > mets;
  iEvent.getByLabel(METSrc_, mets);

  pev_.nMET = 0;

  for (unsigned it=0; it<mets->size();it++) {
    const reco::MET & met = (*mets)[it];
    pev_.METEt[pev_.nMET] = met.et();
    pev_.METPhi[pev_.nMET] = met.phi();
    pev_.METSumEt[pev_.nMET] = met.sumEt();
    pev_.nMET++;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void
METAnalyzer::beginJob()
{

  metTree_ = fs->make<TTree>("metTree","v1");

  // event
  metTree_->Branch("nEv",&pev_.nEv,"nEv/I");
  metTree_->Branch("nLumi",&pev_.nLumi,"nLumi/I");
  metTree_->Branch("nBX",&pev_.nBX,"nBX/I");
  metTree_->Branch("nRun",&pev_.nRun,"nRun/I");

  metTree_->Branch("nMET",&pev_.nMET,"nMET/I");
  metTree_->Branch("MET",pev_.METEt,"MET[nMET]/F");
  metTree_->Branch("METPhi",pev_.METPhi,"METPhi[nMET]/F");
  metTree_->Branch("SumEt",pev_.METSumEt,"SumEt[nMET]/F");
  //
}

// ------------ method called once each job just after ending the event loop  ------------
void
METAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(METAnalyzer);
