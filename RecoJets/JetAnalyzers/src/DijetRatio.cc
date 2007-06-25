// -*- C++ -*-
//
// Package:    DijetRatio
// Class:      DijetRatio
// 
/**\class DijetRatio DijetRatio.cc RecoJets/DijetRatio/src/DijetRatio.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Manoj Jha
//         Created:  Thu Apr 12 15:04:37 CDT 2007
// $Id$
//
//

#include "RecoJets/JetAnalyzers/interface/DijetRatio.h"

DijetRatio::DijetRatio(const edm::ParameterSet& iConfig)
{
   
   //get name of output file with histograms
  fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile"); 
  
  // get names of modules, producing object collections
  m_Mid5CorRecJetsSrc   = iConfig.getParameter<string>("Mid5CorRecJets");
  m_Mid5GenJetsSrc   = iConfig.getParameter<string>("Mid5GenJets");
  m_Mid5CaloJetsSrc   = iConfig.getParameter<string>("Mid5CaloJets");
  
  // eta limit for numerator and denominator
  m_eta3   = iConfig.getParameter<double>("v_etaInner");
  m_eta4   = iConfig.getParameter<double>("v_etaOuter");

}


DijetRatio::~DijetRatio()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DijetRatio::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
   // get calo jet collection
   Handle<CaloJetCollection> Mid5CorRecJets;
   iEvent.getByLabel(m_Mid5CorRecJetsSrc, Mid5CorRecJets);
   
   Handle<CaloJetCollection> Mid5CaloJets;
   iEvent.getByLabel(m_Mid5CaloJetsSrc, Mid5CaloJets);

   Handle<GenJetCollection> Mid5GenJets;
   iEvent.getByLabel(m_Mid5GenJetsSrc, Mid5GenJets);
   
   //etaRange
   /*
   float eta1 = -1.0;
   float eta2 = -0.5;
   float eta3 = 0.5;
   float eta4 = 1.0;
   */

   
   double eta3 = m_eta3;
   double eta4 = m_eta4;
   double eta1 = -eta4;
   double eta2 = -eta3;

   histoFill(hGen, Mid5GenJets, eta1, eta2, eta3, eta4);
   histoFill(hCalo, Mid5CaloJets, eta1, eta2, eta3, eta4);
   histoFill(hCor, Mid5CorRecJets, eta1, eta2, eta3, eta4);
}


// ------------ method called once each job just before starting event loop  ------------
void 
DijetRatio::beginJob(const edm::EventSetup&)
{
   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;

   //Histo title
   char m_hCor[20] = "hCor";
   char m_hCalo[20] = "hCalo";
   char m_hGen[20] = "hGen";
   
   // Histo Initializations for  Jets
   hInit(hGen, m_hGen);
   hInit(hCalo, m_hCalo);
   hInit(hCor, m_hCor);

   return;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DijetRatio::endJob() {
hOutputFile->Write() ;
hOutputFile->Close() ;
return ;
}

