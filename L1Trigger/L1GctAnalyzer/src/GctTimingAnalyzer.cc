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
// $Id: GctTimingAnalyzer.cc,v 1.3 2008/05/13 20:13:41 tapper Exp $
//
//

// Include file
#include "L1Trigger/L1GctAnalyzer/interface/GctTimingAnalyzer.h"

GctTimingAnalyzer::GctTimingAnalyzer(const edm::ParameterSet& iConfig):
  m_outputFileName(iConfig.getUntrackedParameter<std::string>("outFile", "gctTiming.txt")),
  m_isoEmSource(iConfig.getUntrackedParameter<edm::InputTag>("isoEmSource")),
  m_nonIsoEmSource(iConfig.getUntrackedParameter<edm::InputTag>("nonIsoEmSource")),
  m_internEmSource(iConfig.getUntrackedParameter<edm::InputTag>("internEmSource")),
  m_cenJetsSource(iConfig.getUntrackedParameter<edm::InputTag>("cenJetsSource")),
  m_forJetsSource(iConfig.getUntrackedParameter<edm::InputTag>("forJetsSource")),
  m_tauJetsSource(iConfig.getUntrackedParameter<edm::InputTag>("tauJetsSource")),
  m_eSumsSource(iConfig.getUntrackedParameter<edm::InputTag>("eSumsSource")),
  m_fibreSource(iConfig.getUntrackedParameter<edm::InputTag>("fibreSource")),
  m_rctSource(iConfig.getUntrackedParameter<edm::InputTag>("rctSource")),
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

  // Isolated EM cands in GCT output
  Handle<L1GctEmCandCollection> isoEm; 
  iEvent.getByLabel(m_isoEmSource,isoEm);    

  for (L1GctEmCandCollection::const_iterator em=isoEm->begin(); em!=isoEm->end(); em++){
    if (em->rank()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*em) << std::endl; 
    }
  }
  
  // Non-Isolated EM cands in GCT output
  Handle<L1GctEmCandCollection> nonIsoEm; 
  iEvent.getByLabel(m_nonIsoEmSource,nonIsoEm);    

  for (L1GctEmCandCollection::const_iterator em=nonIsoEm->begin(); em!=nonIsoEm->end(); em++){
    if (em->rank()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*em) << std::endl; 
    }
  }

  // Internal GCT EM cands
  Handle<L1GctInternEmCandCollection> internEm; 
  iEvent.getByLabel(m_internEmSource,internEm);    

  for (L1GctInternEmCandCollection::const_iterator em=internEm->begin(); em!=internEm->end(); em++){
    if (em->rank()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*em) << std::endl; 
    }
  }

  // RCT EM cands
  Handle<L1CaloEmCollection> rctEm; 
  iEvent.getByLabel(m_rctSource,rctEm);    

  for (L1CaloEmCollection::const_iterator em=rctEm->begin(); em!=rctEm->end(); em++){
    if (em->rank()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*em) << std::endl; 
    }
  }

  // Central jet cands in GCT output
  Handle<L1GctJetCandCollection> cenJets; 
  iEvent.getByLabel(m_cenJetsSource,cenJets);    

  for (L1GctJetCandCollection::const_iterator cj=cenJets->begin(); cj!=cenJets->end(); cj++){
    if (cj->rank()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*cj) << std::endl; 
    }
  }

  // Forward jet cands in GCT output
  Handle<L1GctJetCandCollection> forJets; 
  iEvent.getByLabel(m_forJetsSource,forJets);    

  for (L1GctJetCandCollection::const_iterator fj=forJets->begin(); fj!=forJets->end(); fj++){
    if (fj->rank()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*fj) << std::endl; 
    }
  }

  // Tau jet cands in GCT output
  Handle<L1GctJetCandCollection> tauJets; 
  iEvent.getByLabel(m_tauJetsSource,tauJets);    

  for (L1GctJetCandCollection::const_iterator tj=tauJets->begin(); tj!=tauJets->end(); tj++){
    if (tj->rank()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*tj) << std::endl; 
    }
  }

  // Missing Et
  Handle<L1GctEtMissCollection> missEt;
  iEvent.getByLabel(m_eSumsSource,missEt);

  for (L1GctEtMissCollection::const_iterator met=missEt->begin(); met!=missEt->end(); met++){
    if (met->et()>0){
      m_outputFile << "BX = " << m_evtNum << " " << (*met) << std::endl; 
    }
  }

  // Total Et
  Handle<L1GctEtTotalCollection> totEt;
  iEvent.getByLabel(m_eSumsSource,totEt);

  for (L1GctEtTotalCollection::const_iterator tet=totEt->begin(); tet!=totEt->end(); tet++){
    if (tet->et()>0){
      m_outputFile << "BX = " << m_evtNum << " " << (*tet) << std::endl; 
    }
  }

  // Ht
  Handle<L1GctEtHadCollection> hadEt;
  iEvent.getByLabel(m_eSumsSource,hadEt);

  for (L1GctEtHadCollection::const_iterator ht=hadEt->begin(); ht!=hadEt->end(); ht++){
    if (ht->et()>0){
      m_outputFile << "BX = " << m_evtNum << " " << (*ht) << std::endl; 
    }
  }

  // Jet counts
  Handle<L1GctJetCountsCollection> jetCnts;
  iEvent.getByLabel(m_eSumsSource,jetCnts);

  for (L1GctJetCountsCollection::const_iterator jc=jetCnts->begin(); jc!=jetCnts->end(); jc++){
    if (jc->raw0() || jc->raw1()){
      m_outputFile << "BX = " << m_evtNum << " " << (*jc) << std::endl; 
    }
  }

  // RCT regions
  Handle<L1CaloRegionCollection> rctRn; 
  iEvent.getByLabel(m_rctSource,rctRn);    

  for (L1CaloRegionCollection::const_iterator rn=rctRn->begin(); rn!=rctRn->end(); rn++){
    if (rn->et()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*rn) << std::endl; 
    }
  }

  m_evtNum++;
}
