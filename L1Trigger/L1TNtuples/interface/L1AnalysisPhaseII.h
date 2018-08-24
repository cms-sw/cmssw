#ifndef __L1Analysis_L1AnalysisPhaseII_H__
#define __L1Analysis_L1AnalysisPhaseII_H__

//-------------------------------------------------------------------------------
// Created 02/03/2010 - A.C. Le Bihan
// 
// 
// Original code : UserCode/L1TriggerDPG/L1ExtraTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

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

#include "DataFormats/L1TVertex/interface/Vertex.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkGlbMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkGlbMuonParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticleFwd.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMET.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisPhaseIIDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisPhaseII 
  {
  public:
    L1AnalysisPhaseII();
    ~L1AnalysisPhaseII();
    void Reset() {l1extra_.Reset();}

    // Fill DZ of Vertex, different algorithms
    void SetVertices(float z0Puppi, float z0VertexTDR, const edm::Handle<std::vector<l1t::Vertex> > l1vertices, const edm::Handle<std::vector<l1t::L1TkPrimaryVertex> > l1TkPrimaryVertex);


    // Old style objects (Phase I)
    void SetTau  (const edm::Handle<l1t::TauBxCollection>    tau,  unsigned maxL1Extra);
    void SetJet  (const edm::Handle<l1t::JetBxCollection>    jet,  unsigned maxL1Extra);
    void SetSum  (const edm::Handle<l1t::EtSumBxCollection>  sums, unsigned maxL1Extra);
    void SetMuon (const edm::Handle<l1t::MuonBxCollection>   muon, unsigned maxL1Extra);

    // Add new standalone objects 
    void SetEG   (const edm::Handle<l1t::EGammaBxCollection> EG,   unsigned maxL1Extra);
    void SetMuonKF (const edm::Handle<l1t::RegionalMuonCandBxCollection>   muonKF, unsigned maxL1Extra);

    // Add L1TrackTriggerObjects
    void SetTkEG   (const  edm::Handle<l1t::L1TkElectronParticleCollection>   tkEG,     unsigned maxL1Extra);
    void SetTkEGLoose   (const  edm::Handle<l1t::L1TkElectronParticleCollection>   tkEGLoose,     unsigned maxL1Extra);
    void SetTkEM   (const  edm::Handle<l1t::L1TkEmParticleCollection>   tkEM,     unsigned maxL1Extra);
    void SetTkGlbMuon (const  edm::Handle<l1t::L1TkGlbMuonParticleCollection> TkGlbMuon,   unsigned maxL1Extra);
    void SetTkMuon (const  edm::Handle<l1t::L1TkMuonParticleCollection> TkMuon,   unsigned maxL1Extra);

    void SetTkTau  (const  edm::Handle<l1t::L1TkTauParticleCollection> tkTau, unsigned maxL1Extra);

    void SetTkJet  (const  edm::Handle<l1t::L1TkJetParticleCollection>  tkTrackerJet,    unsigned maxL1Extra);
    void SetTkCaloJet  (const  edm::Handle<l1t::L1TkJetParticleCollection>  tkCaloJet,    unsigned maxL1Extra);
    void SetTkMET  (const  edm::Handle<l1t::L1TkEtMissParticleCollection> trackerMets);
    void SetTkMHT  (const  edm::Handle<l1t::L1TkHTMissParticleCollection> trackerMHTs);

    // Add new PFJet collections 
    void SetPFJet  (const      edm::Handle<reco::PFJetCollection>  PFJet,    unsigned maxL1Extra);
    void SetL1METPF(const edm::Handle< std::vector<reco::PFMET> > l1MetPF);

    L1AnalysisPhaseIIDataFormat * getData() {return &l1extra_;}




  private :
    L1AnalysisPhaseIIDataFormat l1extra_;
  }; 
}
#endif


