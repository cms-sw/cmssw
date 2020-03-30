/*! \brief   Definition of all the relevant data types
 *  \details Herw we declare instances of all the relevant types. 
 *
 *  \author Vladimir Rekovic
 *  \date   2020, Mar 26 
 *
 */

#ifndef PHASE2L1CORRELATOR_CLASSES_H
#define PHASE2L1CORRELATOR_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
//#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// includes needed for the L1TrackTriggerObjects

#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMissFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkEGTau.h"
#include "DataFormats/L1TCorrelator/interface/L1TrkTau.h"
#include "DataFormats/L1TCorrelator/interface/L1CaloTkTau.h"

#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkHTMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMissFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkGlbMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkGlbMuonFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkTau.h"
#include "DataFormats/L1TCorrelator/interface/TkTauFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkPhiCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidateFwd.h"

#include "DataFormats/L1TCorrelator/interface/TkBsCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkBsCandidateFwd.h"

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace {

  //  Phase2 L1 Correlator Objects :

  struct dictionaryl1tkobj {
    // L1 Primary Vertex
    l1t::TkPrimaryVertex trzvtx;
    edm::Wrapper<l1t::TkPrimaryVertexCollection> trzvtxColl;
    edm::Ref<l1t::TkPrimaryVertexCollection> trkvtxRef;

    // TkEtMiss... following L1EtMiss...
    l1t::TkEtMiss TketMiss;
    l1t::TkEtMissCollection TketMissColl;
    edm::Wrapper<l1t::TkEtMiss> w_TketMiss;
    edm::Wrapper<l1t::TkEtMissCollection> w_TketMissColl;
    //l1t::TkEtMissRef refTkEtMiss ;
    //l1t::TkEtMissRefVector refTkVecEtMiss ;
    //l1t::TkEtMissVectorRef vecTkRefEtMissColl ;
    //l1t::TkEtMissRefProd refTkProdEtMiss ;
    //edm::reftobase::Holder<reco::Candidate, l1t::TkEtMissRef> rtbTkm1;
    //edm::reftobase::Holder<reco::Candidate, l1t::TkEtMissRefProd> rtbTkm2;

    // TkEm
    l1t::TkEmCollection trkemColl;
    edm::Wrapper<l1t::TkEmCollection> w_trkemColl;
    l1t::TkEmRef reftrkEm;
    //l1t::TkEmRefVector refVectrkEmColl ;
    //l1t::TkEmVectorRef vecReftrkEmColl ;
    //edm::reftobase::Holder<reco::Candidate, l1t::TkEmRef> rtbtrke;

    // TkEGTau
    l1t::TkEGTauCollection trkegColl;
    edm::Wrapper<l1t::TkEGTauCollection> w_trkegColl;
    l1t::TkEGTauRef reftrkEG;

    // L1TrkTau
    l1t::L1TrkTauCollection tktauColl;
    edm::Wrapper<l1t::L1TrkTauCollection> w_tktauColl;
    l1t::L1TrkTauRef reftkTau;

    // L1CaloTkTau
    l1t::L1CaloTkTauCollection calotrktauColl;
    edm::Wrapper<l1t::L1CaloTkTauCollection> w_calotrktauColl;
    l1t::L1CaloTkTauRef refcalotrkTau;

    // TkElectron
    l1t::TkElectronCollection trkeleColl;
    edm::Wrapper<l1t::TkElectronCollection> w_trkeleColl;
    l1t::TkElectronRef reftrkEle;

    // TkJet
    l1t::TkJetCollection trkjetColl;
    edm::Wrapper<l1t::TkJetCollection> w_trkjetColl;
    l1t::TkJetRef reftrkJet;
    l1t::TkJetRefProd refTkProdJet;

    // TkHTMiss
    l1t::TkHTMiss TkHTMiss;
    l1t::TkHTMissCollection TkHTMissColl;
    edm::Wrapper<l1t::TkHTMiss> w_TkHTMiss;
    edm::Wrapper<l1t::TkHTMissCollection> w_TkHTMissColl;

    // TkMuon
    l1t::TkMuonCollection trkmuColl;
    edm::Wrapper<l1t::TkMuonCollection> w_trkmuColl;
    l1t::TkMuonRef reftrkMu;

    // TkGlbMuon
    l1t::TkGlbMuonCollection trkglbmuColl;
    edm::Wrapper<l1t::TkGlbMuonCollection> w_trkglbmuColl;
    l1t::TkGlbMuonRef reftrkGlbMu;

    // TkTau
    l1t::TkTauCollection trktauColl;
    edm::Wrapper<l1t::TkTauCollection> w_trktauColl;
    l1t::TkTauRef reftrkTau;

    // TkPhiCandidate
    l1t::TkPhiCandidateCollection trkPhiColl;
    edm::Wrapper<l1t::TkPhiCandidateCollection> w_trkPhiColl;
    l1t::TkPhiCandidateRef reftrkPhi;
    //l1t::TkPhiCandidateRefProd refTkProdPhi;

    // TkBsCandidate
    l1t::TkBsCandidateCollection trkBsColl;
    edm::Wrapper<l1t::TkBsCandidateCollection> w_trkBsColl;
    l1t::TkBsCandidateRef reftrkBs;
    //l1t::TkBsCandidateRefProd refTkProdBs;
  };
}  // namespace

#endif
