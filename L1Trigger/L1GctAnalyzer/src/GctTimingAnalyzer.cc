// -*- C++ -*-
//
// Package:    GctTimingAnalyzer
// Class:      GctTimingAnalyzer
// 
/**\class GctTimingAnalyzer GctTimingAnalyzer.cc L1Trigger/L1GctAnalzyer/src/GctTimingAnalyzer.cc

Description: Analyse the timing of all of the GCT pipelines

*/
//
// Original Author:  Alex Tapper
//         Created:  Mon Apr 21 14:21:06 CEST 2008
// $Id: GctTimingAnalyzer.cc,v 1.9 2008/11/05 20:34:17 tapper Exp $
//
//

// Include file
#include "L1Trigger/L1GctAnalyzer/interface/GctTimingAnalyzer.h"

GctTimingAnalyzer::GctTimingAnalyzer(const edm::ParameterSet& iConfig):
  m_outputFileName(iConfig.getUntrackedParameter<std::string>("outFile", "gctTiming.txt")),
  m_gctSource(iConfig.getUntrackedParameter<edm::InputTag>("gctSource")),
  m_isoEmSource(iConfig.getUntrackedParameter<edm::InputTag>("isoEmSource")),
  m_nonIsoEmSource(iConfig.getUntrackedParameter<edm::InputTag>("nonIsoEmSource")),
  m_cenJetsSource(iConfig.getUntrackedParameter<edm::InputTag>("cenJetsSource")),
  m_forJetsSource(iConfig.getUntrackedParameter<edm::InputTag>("forJetsSource")),
  m_tauJetsSource(iConfig.getUntrackedParameter<edm::InputTag>("tauJetsSource")),
  m_doInternal(iConfig.getUntrackedParameter<bool>("doInternal")),
  m_doElectrons(iConfig.getUntrackedParameter<bool>("doElectrons")),
  m_doJets(iConfig.getUntrackedParameter<bool>("doJets")),
  m_doHFRings(iConfig.getUntrackedParameter<bool>("doHFRings")),
  m_doESums(iConfig.getUntrackedParameter<bool>("doESums")),
  m_evtNum(0)
{
  m_outputFile.open(m_outputFileName.c_str());
}

GctTimingAnalyzer::~GctTimingAnalyzer()
{
  m_outputFile.close();
}

void GctTimingAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  
  // Electrons
  if(m_doElectrons){

    // Isolated EM cands in GCT output
    Handle<L1GctEmCandCollection> isoEm; 
    iEvent.getByLabel(m_isoEmSource,isoEm);    

    for (L1GctEmCandCollection::const_iterator em=isoEm->begin(); em!=isoEm->end(); em++){
      if (em->rank()>0) {
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*em) << std::endl; 
      }
    }
  
    // Non-Isolated EM cands in GCT output
    Handle<L1GctEmCandCollection> nonIsoEm; 
    iEvent.getByLabel(m_nonIsoEmSource,nonIsoEm);    
    
    for (L1GctEmCandCollection::const_iterator em=nonIsoEm->begin(); em!=nonIsoEm->end(); em++){
      if (em->rank()>0) {
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*em) << std::endl; 
      }
    }

    if (m_doInternal){
      // Internal GCT EM cands
      Handle<L1GctInternEmCandCollection> internEm; 
      iEvent.getByLabel(m_gctSource,internEm);    

      if (internEm.isValid()) {
        for (L1GctInternEmCandCollection::const_iterator em=internEm->begin(); em!=internEm->end(); em++){
          if (em->rank()>0) {
            m_outputFile << "BX = " << dec << m_evtNum << " " << (*em) << std::endl; 
          }
        }
      }
    }

    // RCT EM cands
    Handle<L1CaloEmCollection> rctEm; 
    iEvent.getByLabel(m_gctSource,rctEm);    
    
    for (L1CaloEmCollection::const_iterator em=rctEm->begin(); em!=rctEm->end(); em++){
      if (em->rank()>0) {
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*em) << std::endl; 
      }
    }
  }     
  
  // Jets
  if (m_doJets){

    // Central jet cands in GCT output
    Handle<L1GctJetCandCollection> cenJets; 
    iEvent.getByLabel(m_cenJetsSource,cenJets);    

    for (L1GctJetCandCollection::const_iterator cj=cenJets->begin(); cj!=cenJets->end(); cj++){
      if (cj->rank()>0) {
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*cj) << std::endl; 
      }
    }
    
    // Forward jet cands in GCT output
    Handle<L1GctJetCandCollection> forJets; 
    iEvent.getByLabel(m_forJetsSource,forJets);    
    
    for (L1GctJetCandCollection::const_iterator fj=forJets->begin(); fj!=forJets->end(); fj++){
      if (fj->rank()>0) {
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*fj) << std::endl; 
      }
    }
    
    // Tau jet cands in GCT output
    Handle<L1GctJetCandCollection> tauJets; 
    iEvent.getByLabel(m_tauJetsSource,tauJets);    
    
    for (L1GctJetCandCollection::const_iterator tj=tauJets->begin(); tj!=tauJets->end(); tj++){
      if (tj->rank()>0) {
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*tj) << std::endl; 
      }
    }
  
    if (m_doInternal){

      // Internal GCT jet cands
      Handle<L1GctInternJetDataCollection> internJets; 
      iEvent.getByLabel(m_gctSource,internJets);    
    
      if (internJets.isValid()){
        for (L1GctInternJetDataCollection::const_iterator j=internJets->begin(); j!=internJets->end(); j++){
          if ((j->et()>0) || (j->rank()>0)) {
            m_outputFile << "BX = " << dec << m_evtNum << " " << (*j) << std::endl; 
          }
        }
      }
    }
  }
  
  // RCT regions
  Handle<L1CaloRegionCollection> rctRn; 
  iEvent.getByLabel(m_gctSource,rctRn);    
    
  for (L1CaloRegionCollection::const_iterator rn=rctRn->begin(); rn!=rctRn->end(); rn++){
    if (rn->et()>0) {
      m_outputFile << "BX = " << dec << m_evtNum << " " << (*rn) << std::endl; 
    }
  }

  // HF Rings
  if (m_doHFRings){

    // HFRing counts
    Handle<L1GctHFBitCountsCollection> hfBitCnt;
    iEvent.getByLabel(m_gctSource,hfBitCnt);

    for (L1GctHFBitCountsCollection::const_iterator jc=hfBitCnt->begin(); jc!=hfBitCnt->end(); jc++){
      if (jc->bitCount(0) || jc->bitCount(1) || jc->bitCount(2) || jc->bitCount(3)){
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*jc) << std::endl; 
      }
    }

    // HFRing Et sums
    Handle<L1GctHFRingEtSumsCollection> hfEtSums;
    iEvent.getByLabel(m_gctSource,hfEtSums);

    for (L1GctHFRingEtSumsCollection::const_iterator js=hfEtSums->begin(); js!=hfEtSums->end(); js++){
      if (js->etSum(0) || js->etSum(1) || js->etSum(2) || js->etSum(3)){
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*js) << std::endl; 
      }
    }

    if (m_doInternal){
      // Internal HF data
      Handle<L1GctInternHFDataCollection> internHF;
      iEvent.getByLabel(m_gctSource,internHF);
      
      if (internHF.isValid()){
        for (L1GctInternHFDataCollection::const_iterator hf=internHF->begin(); hf!=internHF->end(); hf++){
          if (hf->value(0) || hf->value(1) || hf->value(2) || hf->value(3)){
            m_outputFile << "BX = " << dec << m_evtNum << " " << (*hf) << std::endl; 
          }
        }
      }
    }    
  }

  // HT, MET and ET
  if (m_doESums){

    // MET
    Handle<L1GctEtMissCollection> missEt;
    iEvent.getByLabel(m_gctSource,missEt);

    for (L1GctEtMissCollection::const_iterator met=missEt->begin(); met!=missEt->end(); met++){
      if (met->et()>0){
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*met) << std::endl; 
      }
    }

    // Total Et
    Handle<L1GctEtTotalCollection> totEt;
    iEvent.getByLabel(m_gctSource,totEt);
    
    for (L1GctEtTotalCollection::const_iterator tet=totEt->begin(); tet!=totEt->end(); tet++){
      if (tet->et()>0){
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*tet) << std::endl; 
      }
    }
    
    // Ht
    Handle<L1GctEtHadCollection> hadEt;
    iEvent.getByLabel(m_gctSource,hadEt);

    for (L1GctEtHadCollection::const_iterator ht=hadEt->begin(); ht!=hadEt->end(); ht++){
      if (ht->et()>0){
        m_outputFile << "BX = " << dec << m_evtNum << " " << (*ht) << std::endl; 
      }
    }

    if (m_doInternal){
      // Internal GCT Et sums
      Handle<L1GctInternEtSumCollection> Et;
      iEvent.getByLabel(m_gctSource,Et);
      
      if (Et.isValid()){
        for (L1GctInternEtSumCollection::const_iterator e=Et->begin(); e!=Et->end(); e++){
          if (e->et()>0){
            m_outputFile << "BX = " << dec << m_evtNum << " " << (*e) << std::endl;
          }
        }
      }
    }    
  }

  m_evtNum++;

}
