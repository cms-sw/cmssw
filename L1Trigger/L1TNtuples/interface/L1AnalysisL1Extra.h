#ifndef __L1Analysis_L1AnalysisL1Extra_H__
#define __L1Analysis_L1AnalysisL1Extra_H__

//-------------------------------------------------------------------------------
// Created 02/03/2010 - A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1ExtraTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "L1AnalysisL1ExtraDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisL1Extra 
  {
  public:
    L1AnalysisL1Extra();
    ~L1AnalysisL1Extra();
    void Reset() {l1extra_.Reset();}
    void SetIsoEm   (const edm::Handle<l1extra::L1EmParticleCollection>     isoEm,    unsigned maxL1Extra);
    void SetNonIsoEm(const edm::Handle<l1extra::L1EmParticleCollection>     nonIsoEm, unsigned maxL1Extra);
    void SetCenJet  (const edm::Handle<l1extra::L1JetParticleCollection>    cenJet,   unsigned maxL1Extra);
    void SetFwdJet  (const edm::Handle<l1extra::L1JetParticleCollection>    fwdJet,   unsigned maxL1Extra);
    void SetTauJet  (const edm::Handle<l1extra::L1JetParticleCollection>    tauJet,   unsigned maxL1Extra);
    void SetIsoTauJet(const edm::Handle<l1extra::L1JetParticleCollection>   isoTauJet,unsigned maxL1Extra);
    void SetMuon    (const edm::Handle<l1extra::L1MuonParticleCollection>   muon,     unsigned maxL1Extra);
    void SetMet     (const edm::Handle<l1extra::L1EtMissParticleCollection> mets);
    void SetMht     (const edm::Handle<l1extra::L1EtMissParticleCollection> mhts);
    void SetHFring  (const edm::Handle<l1extra::L1HFRingsCollection>        hfRings);
    L1AnalysisL1ExtraDataFormat * getData() {return &l1extra_;}

  private :
    L1AnalysisL1ExtraDataFormat l1extra_;
  }; 
}
#endif


