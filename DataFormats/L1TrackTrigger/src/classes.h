/*! \brief   Definition of all the relevant data types
 *  \details Herw we declare instances of all the relevant types. 
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTPixelTrack.h"

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

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"



namespace
{
  namespace
  {
    /// Main template type
    Ref_PixelDigi_  PD;

    /// TTCluster and containers
    TTCluster< Ref_PixelDigi_ >                                               C_PD;
    std::vector< TTCluster< Ref_PixelDigi_ > >                              V_C_PD;
    edm::Wrapper< std::vector< TTCluster< Ref_PixelDigi_ > > >            W_V_C_PD;
    edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >                   SDV_C_PD;
    edm::Wrapper< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > W_SDV_C_PD;

    /// edm::Ref to TTCluster in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > >                                    R_C_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > >                  W_R_C_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > >                   V_R_C_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > > > W_V_R_C_PD;

    /// TTStub and containers
    TTStub< Ref_PixelDigi_ >                                               S_PD;
    std::vector< TTStub< Ref_PixelDigi_ > >                              V_S_PD;
    edm::Wrapper< std::vector< TTStub< Ref_PixelDigi_ > > >            W_V_S_PD;
    edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >                   SDV_S_PD;
    edm::Wrapper< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > W_SDV_S_PD;

    /// edm::Ref to TTStub in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > >                                    R_S_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >                  W_R_S_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >                   V_R_S_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > > W_V_R_S_PD;

    /// TTTrack and containers
    TTTrack< Ref_PixelDigi_ >                                    T_PD;
    std::vector< TTTrack< Ref_PixelDigi_ > >                   V_T_PD;
    edm::Wrapper< std::vector< TTTrack< Ref_PixelDigi_ > > > W_V_T_PD;

    /// edm::Ptr to TTTrack and containers
    edm::Ptr< TTTrack< Ref_PixelDigi_ > >                                    P_T_PD;
    edm::Wrapper< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >                  W_P_T_PD;
    std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >                   V_P_T_PD;
    edm::Ref< std::vector < TTTrack< Ref_PixelDigi_ > > >                   R_V_T_PD;
    edm::Wrapper< std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > > W_V_P_T_PD;

    /// TTPixelTrack and containers
    TTPixelTrack                                                TT_PT;
    std::vector< TTPixelTrack >                               V_TT_PT;
    edm::Wrapper< std::vector< TTPixelTrack > >             W_V_TT_PT;

    /// edm::Ptr to TTPixelTrack and containers
    edm::Ptr< TTPixelTrack >                                    P_TT_PT;
    edm::Wrapper< edm::Ptr< TTPixelTrack > >                  W_P_TT_PT;
    std::vector< edm::Ptr< TTPixelTrack > >                   V_P_TT_PT;
    edm::Wrapper< std::vector< edm::Ptr< TTPixelTrack > > > W_V_P_TT_PT;


  }
}



//  L1TrackTriggerObjects stuff :

namespace {
  struct dictionary {


        // L1 Primary Vertex
     L1TkPrimaryVertex trzvtx;
     edm::Wrapper<L1TkPrimaryVertexCollection> trzvtxColl;
     edm::Ref< L1TkPrimaryVertexCollection > trkvtxRef ;


        // L1TkEtMiss... following L1EtMiss...
     l1extra::L1TkEtMissParticle TketMiss ;
     l1extra::L1TkEtMissParticleCollection TketMissColl ;
     edm::Wrapper<l1extra::L1TkEtMissParticle> w_TketMiss;
     edm::Wrapper<l1extra::L1TkEtMissParticleCollection> w_TketMissColl;
     //l1extra::L1TkEtMissParticleRef refTkEtMiss ;
     //l1extra::L1TkEtMissParticleRefVector refTkVecEtMiss ;
     //l1extra::L1TkEtMissParticleVectorRef vecTkRefEtMissColl ;
     //l1extra::L1TkEtMissParticleRefProd refTkProdEtMiss ;
     //edm::reftobase::Holder<reco::Candidate, l1extra::L1TkEtMissParticleRef> rtbTkm1;
     //edm::reftobase::Holder<reco::Candidate, l1extra::L1TkEtMissParticleRefProd> rtbTkm2;
        
        // L1TkEmParticle
     l1extra::L1TkEmParticleCollection trkemColl ;
     edm::Wrapper<l1extra::L1TkEmParticleCollection> w_trkemColl;
     l1extra::L1TkEmParticleRef reftrkEm ;
     //l1extra::L1TkEmParticleRefVector refVectrkEmColl ;
     //l1extra::L1TkEmParticleVectorRef vecReftrkEmColl ;
     //edm::reftobase::Holder<reco::Candidate, l1extra::L1TkEmParticleRef> rtbtrke;
        
        // L1TkElectronParticle
     l1extra::L1TkElectronParticleCollection trkeleColl ;
     edm::Wrapper<l1extra::L1TkElectronParticleCollection> w_trkeleColl;
     l1extra::L1TkElectronParticleRef reftrkEle ;
        
        // L1TkJetParticle
     l1extra::L1TkJetParticleCollection trkjetColl ;
     edm::Wrapper<l1extra::L1TkJetParticleCollection> w_trkjetColl;
     l1extra::L1TkJetParticleRef reftrkJet ;
     l1extra::L1TkJetParticleRefProd refTkProdJet ;

        // L1TkHTMissParticle
     l1extra::L1TkHTMissParticle TkHTMiss ;
     l1extra::L1TkHTMissParticleCollection TkHTMissColl ;
     edm::Wrapper<l1extra::L1TkHTMissParticle> w_TkHTMiss;
     edm::Wrapper<l1extra::L1TkHTMissParticleCollection> w_TkHTMissColl;
        
        // L1TkMuonParticle
     l1extra::L1TkMuonParticleCollection trkmuColl ;
     edm::Wrapper<l1extra::L1TkMuonParticleCollection> w_trkmuColl;
     l1extra::L1TkMuonParticleRef reftrkMu ;
        
        // L1TkTauParticle
     l1extra::L1TkTauParticleCollection trktauColl ;
     edm::Wrapper<l1extra::L1TkTauParticleCollection> w_trktauColl;
     l1extra::L1TkTauParticleRef reftrkTau ;

  
  };

}


