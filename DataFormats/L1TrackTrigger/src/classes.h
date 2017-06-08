/*! \brief   Definition of all the relevant data types
 *  \details Herw we declare instances of all the relevant types. 
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1TRACKTRIGGER_CLASSES_H
#define L1TRACKTRIGGER_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// includes needed for the L1TrackTriggerObjects

#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticleFwd.h"

//#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
//#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace
{
  struct dictionary1 {
    /// Main template type
    Ref_Phase2TrackerDigi_  PD;
    std::vector< Ref_Phase2TrackerDigi_ >                              V_PD;
  };

  struct dictionary2 {
    /// TTCluster and containers
    TTCluster< Ref_Phase2TrackerDigi_ >                                               C_PD;
    std::vector< TTCluster< Ref_Phase2TrackerDigi_ > >                              V_C_PD;
    edm::Wrapper< std::vector< TTCluster< Ref_Phase2TrackerDigi_ > > >            W_V_C_PD;
    edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >                   SDV_C_PD;
    edm::Wrapper< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > > > W_SDV_C_PD;

    /// edm::Ref to TTCluster in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > >                                    R_C_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > >                  W_R_C_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > >                   V_R_C_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > > > W_V_R_C_PD;
  };

  struct dictionary3 {
    /// TTStub and containers
    TTStub< Ref_Phase2TrackerDigi_ >                                               S_PD;
    std::vector< TTStub< Ref_Phase2TrackerDigi_ > >                              V_S_PD;
    edm::Wrapper< std::vector< TTStub< Ref_Phase2TrackerDigi_ > > >            W_V_S_PD;
    edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >                   SDV_S_PD;
    edm::Wrapper< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > > W_SDV_S_PD;

    /// edm::Ref to TTStub in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > >                                    R_S_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >                  W_R_S_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >                   V_R_S_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > > > W_V_R_S_PD;
  };


  struct dictionarytrack {
    /// TTTrack and containers
    TTTrack< Ref_Phase2TrackerDigi_ >                                               T_PD;
    std::vector< TTTrack< Ref_Phase2TrackerDigi_ > >                              V_T_PD;
    edm::Wrapper< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >            W_V_T_PD;
    edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >                   SDV_T_PD;
    edm::Wrapper< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > > > W_SDV_T_PD;

    /// edm::Ref to TTTrack in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >, TTTrack< Ref_Phase2TrackerDigi_ > >                                    R_T_PD;
    edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > >                                                                                               P_T_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >, TTTrack< Ref_Phase2TrackerDigi_ > > >                  W_R_T_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >, TTTrack< Ref_Phase2TrackerDigi_ > > >                   V_R_T_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >, TTTrack< Ref_Phase2TrackerDigi_ > > > > W_V_R_T_PD;
  };


//  L1TrackTriggerObjects stuff :

  struct dictionaryl1tkobj {
    // L1 Primary Vertex
    L1TkPrimaryVertex trzvtx;
    edm::Wrapper<L1TkPrimaryVertexCollection> trzvtxColl;
    edm::Ref< L1TkPrimaryVertexCollection > trkvtxRef ;
    
    
    // L1TkEtMiss... following L1EtMiss...
    l1t::L1TkEtMissParticle TketMiss ;
    l1t::L1TkEtMissParticleCollection TketMissColl ;
    edm::Wrapper<l1t::L1TkEtMissParticle> w_TketMiss;
    edm::Wrapper<l1t::L1TkEtMissParticleCollection> w_TketMissColl;
    //l1t::L1TkEtMissParticleRef refTkEtMiss ;
    //l1t::L1TkEtMissParticleRefVector refTkVecEtMiss ;
    //l1t::L1TkEtMissParticleVectorRef vecTkRefEtMissColl ;
    //l1t::L1TkEtMissParticleRefProd refTkProdEtMiss ;
    //edm::reftobase::Holder<reco::Candidate, l1t::L1TkEtMissParticleRef> rtbTkm1;
    //edm::reftobase::Holder<reco::Candidate, l1t::L1TkEtMissParticleRefProd> rtbTkm2;
    
    // L1TkEmParticle
    l1t::L1TkEmParticleCollection trkemColl ;
    edm::Wrapper<l1t::L1TkEmParticleCollection> w_trkemColl;
    l1t::L1TkEmParticleRef reftrkEm ;
    //l1t::L1TkEmParticleRefVector refVectrkEmColl ;
    //l1t::L1TkEmParticleVectorRef vecReftrkEmColl ;
    //edm::reftobase::Holder<reco::Candidate, l1t::L1TkEmParticleRef> rtbtrke;
    
    // L1TkElectronParticle
    l1t::L1TkElectronParticleCollection trkeleColl ;
    edm::Wrapper<l1t::L1TkElectronParticleCollection> w_trkeleColl;
    l1t::L1TkElectronParticleRef reftrkEle ;
    
    // L1TkJetParticle
    l1t::L1TkJetParticleCollection trkjetColl ;
    edm::Wrapper<l1t::L1TkJetParticleCollection> w_trkjetColl;
    l1t::L1TkJetParticleRef reftrkJet ;
    l1t::L1TkJetParticleRefProd refTkProdJet ;
    
    // L1TkHTMissParticle
     l1t::L1TkHTMissParticle TkHTMiss ;
    l1t::L1TkHTMissParticleCollection TkHTMissColl ;
    edm::Wrapper<l1t::L1TkHTMissParticle> w_TkHTMiss;
    edm::Wrapper<l1t::L1TkHTMissParticleCollection> w_TkHTMissColl;
    
    // L1TkMuonParticle
    //     l1t::L1TkMuonParticleCollection trkmuColl ;
    //     edm::Wrapper<l1t::L1TkMuonParticleCollection> w_trkmuColl;
    //     l1t::L1TkMuonParticleRef reftrkMu ;
    
    // L1TkTauParticle
    l1t::L1TkTauParticleCollection trktauColl ;
    edm::Wrapper<l1t::L1TkTauParticleCollection> w_trktauColl;
    l1t::L1TkTauParticleRef reftrkTau ;

    
  };


}

#endif
