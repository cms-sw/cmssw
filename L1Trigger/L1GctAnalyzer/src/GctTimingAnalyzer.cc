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
// $Id: GctTimingAnalyzer.cc,v 1.1 2008/04/21 13:02:12 tapper Exp $
//
//

// Include file
#include "L1Trigger/L1GctAnalyzer/interface/GctTimingAnalyzer.h"

GctTimingAnalyzer::GctTimingAnalyzer(const edm::ParameterSet& iConfig):
  m_outputFileName(iConfig.getUntrackedParameter<std::string>("outFile", "gctTiming.txt")),
  m_isoEmSource(iConfig.getUntrackedParameter<edm::InputTag>("isoEmSource")),
  m_nonIsoEmSource(iConfig.getUntrackedParameter<edm::InputTag>("nonIsoEmSource")),
  m_internEmSource(iConfig.getUntrackedParameter<edm::InputTag>("internEmSource")),
  m_rctEmSource(iConfig.getUntrackedParameter<edm::InputTag>("rctEmSource")),
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
  
  // Non-Iisolated EM cands in GCT output
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
  iEvent.getByLabel(m_rctEmSource,rctEm);    

  for (L1CaloEmCollection::const_iterator em=rctEm->begin(); em!=rctEm->end(); em++){
    if (em->rank()>0) {
      m_outputFile << "BX = " << m_evtNum << " " << (*em) << std::endl; 
    }
  }

  m_evtNum++;
}
