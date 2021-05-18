#ifndef __L1Analysis_L1AnalysisPhaseIIStep1_H__
#define __L1Analysis_L1AnalysisPhaseIIStep1_H__

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

#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkGlbMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkGlbMuonFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMissFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMissFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkTau.h"
#include "DataFormats/L1TCorrelator/interface/TkTauFwd.h"
#include "DataFormats/L1TCorrelator/interface/L1TrkTau.h"
#include "DataFormats/L1TCorrelator/interface/TkEGTau.h"
#include "DataFormats/L1TCorrelator/interface/L1CaloTkTau.h"

//#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"

//#include "DataFormats/Phase2L1Taus/interface/L1HPSPFTau.h"
//#include "DataFormats/Phase2L1Taus/interface/L1HPSPFTauFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkBsCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkBsCandidateFwd.h"

//#include "DataFormats/L1TMuon/interface/BayesMuCorrelatorTrack.h"

#include "DataFormats/JetReco/interface/CaloJet.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisPhaseIIStep1DataFormat.h"

namespace L1Analysis {
  class L1AnalysisPhaseIIStep1 {
  public:
    L1AnalysisPhaseIIStep1();
    ~L1AnalysisPhaseIIStep1();
    void Reset() { l1extra_.Reset(); }

    // Fill DZ of Vertex, different algorithms
    void SetVertices(float z0Puppi, const edm::Handle<std::vector<l1t::TkPrimaryVertex>> TkPrimaryVertex);

    // Add new standalone objects
    void SetEG(const edm::Handle<l1t::EGammaBxCollection> EG,
               const edm::Handle<l1t::EGammaBxCollection> EGHGC,
               unsigned maxL1Extra);
    void SetCaloTau(const edm::Handle<l1t::TauBxCollection> calotau, unsigned maxL1Extra);

    // Add L1TrackTriggerObjects
    void SetTkEG(const edm::Handle<l1t::TkElectronCollection> tkEG,
                 const edm::Handle<l1t::TkElectronCollection> tkEGHGC,
                 unsigned maxL1Extra);
    void SetTkEM(const edm::Handle<l1t::TkEmCollection> tkEM,
                 const edm::Handle<l1t::TkEmCollection> tkEMHGC,
                 unsigned maxL1Extra);

    void SetMuonKF(const edm::Handle<l1t::RegionalMuonCandBxCollection> muonKF,
                   unsigned maxL1Extra,
                   unsigned int muonDetector);
    void SetMuonEMTF(const edm::Handle<l1t::EMTFTrackCollection> muonKF,
                   unsigned maxL1Extra,
                   unsigned int muonDetector);

    void SetTkMuon(const edm::Handle<l1t::TkMuonCollection> TkMuon, unsigned maxL1Extra);

    void SetMuon(const edm::Handle<l1t::MuonBxCollection> muon, unsigned maxL1Extra);

    void SetTkGlbMuon(const edm::Handle<l1t::TkGlbMuonCollection> TkGlbMuon, unsigned maxL1Extra); 

    // Add new PFJet collections
    void SetL1METPF(const edm::Handle<std::vector<reco::PFMET>> l1MetPF);

    // reco::caloJet collection for "Phase1L1Jets" ...
    void SetL1PfPhase1L1TJet(const edm::Handle<std::vector<reco::CaloJet>> l1L1PFPhase1L1Jet, unsigned maxL1Extra);

    void SetPFJet(const edm::Handle<l1t::PFJetCollection> PFJet, unsigned maxL1Extra);

    // Add nntaus
    void SetNNTaus(const edm::Handle<std::vector<l1t::PFTau>> l1nnTaus, unsigned maxL1Extra);

    //tkjets, tkmet, tkht
    void SetTkJet(const edm::Handle<l1t::TkJetCollection> tkTrackerJet, unsigned maxL1Extra);
    void SetTkJetDisplaced(const edm::Handle<l1t::TkJetCollection> tkTrackerJet, unsigned maxL1Extra);

    void SetTkMET(const edm::Handle<l1t::TkEtMissCollection> trackerMets);
    void SetTkMHT(const edm::Handle<l1t::TkHTMissCollection> trackerMHTs);
    void SetTkMETDisplaced(const edm::Handle<l1t::TkEtMissCollection> trackerMets);
    void SetTkMHTDisplaced(const edm::Handle<l1t::TkHTMissCollection> trackerMHTs);


    L1AnalysisPhaseIIStep1DataFormat* getData() { return &l1extra_; }

    static int transverseCoord(double cxa,
                               double cya,
                               double ra,
                               double cxb,
                               double cyb,
                               double rb,
                               double& xg1,
                               double& yg1,
                               double& xg2,
                               double& yg2) dso_internal;

    // Computes z-coordinate on helix at given transverse coordinates
    static double zCoord(const GlobalVector& mom,
                         const GlobalPoint& pos,
                         double r,
                         double xc,
                         double yc,
                         double xg,
                         double yg) dso_internal;

  private:
    L1AnalysisPhaseIIStep1DataFormat l1extra_;
    int tk_nFitParams_ = 4;  // Harcoding this, choosing 4,
        // to not have to store the chosen fitParams for all objects in this tree producer as a configuration.
        // (it would be cleaner if all objects save the Z directly as well as the pointer to the track, or if
        // it is clear that the default is 4 unless specifically stated)
  };
}  // namespace L1Analysis
#endif
