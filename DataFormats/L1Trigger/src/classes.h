#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "Rtypes.h"
#include "Math/Cartesian3D.h"
#include "Math/Polar3D.h"
#include "Math/CylindricalEta3D.h"
#include "Math/PxPyPzE4D.h"
#include <boost/cstdint.hpp>
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h"
#include "DataFormats/L1Trigger/interface/L1DataEmulRecord.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/L1Trigger/interface/L1TriggerError.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/CaloSpare.h"
#include "DataFormats/L1Trigger/interface/L1DataEmulResult.h"

namespace DataFormats_L1Trigger {
  struct dictionary {

    l1t::L1CandidateBxCollection l1CandidateBxColl;
    l1t::EGammaBxCollection eGammaBxColl;
    l1t::EtSumBxCollection  etSumBxColl;
    l1t::JetBxCollection    jetBxColl;
    l1t::MuonBxCollection   muonBxColl;
    l1t::TauBxCollection    tauBxColl;
    l1t::CaloSpareBxCollection caloSpareColl;
    l1t::L1DataEmulResultBxCollection deResult;

    edm::Wrapper<l1t::L1CandidateBxCollection> w_l1CandidateBxColl;
    edm::Wrapper<l1t::EGammaBxCollection> w_eGammaBxColl;
    edm::Wrapper<l1t::EtSumBxCollection>  w_etSumBxColl;
    edm::Wrapper<l1t::JetBxCollection>    w_jetBxColl;
    edm::Wrapper<l1t::MuonBxCollection>   w_muonBxColl;
    edm::Wrapper<l1t::TauBxCollection>    w_tauBxColl;
    edm::Wrapper<l1t::CaloSpareBxCollection> w_caloSpareColl;
    edm::Wrapper<l1t::L1DataEmulResultBxCollection>   w_deResult;

    l1t::MuonRef   refMuon_;
    l1t::MuonRefVector   refVecMuon_;
    l1t::MuonVectorRef   vecRefMuon_;

    l1extra::L1EmParticleCollection emColl ;
    l1extra::L1JetParticleCollection jetColl ;
    l1extra::L1MuonParticleCollection muonColl ;
    l1extra::L1EtMissParticle etMiss ;
    l1extra::L1EtMissParticleCollection etMissColl ;
    l1extra::L1ParticleMapCollection mapColl ;
    l1extra::L1HFRingsCollection hfRingsColl ;

    edm::Wrapper<l1extra::L1EmParticleCollection> w_emColl;
    edm::Wrapper<l1extra::L1JetParticleCollection> w_jetColl;
    edm::Wrapper<l1extra::L1MuonParticleCollection> w_muonColl;
    edm::Wrapper<l1extra::L1EtMissParticle> w_etMiss;
    edm::Wrapper<l1extra::L1EtMissParticleCollection> w_etMissColl;
    edm::Wrapper<l1extra::L1ParticleMapCollection> w_mapColl;
    edm::Wrapper<l1extra::L1HFRingsCollection> w_hfRingsColl;

    l1extra::L1EmParticleRef refEm ;
    l1extra::L1JetParticleRef refJet ;
    l1extra::L1MuonParticleRef refMuon ;
    l1extra::L1EtMissParticleRef refEtMiss ;
    l1extra::L1HFRingsRef refHFRings ;

    l1extra::L1EmParticleRefVector refVecEmColl ;
    l1extra::L1JetParticleRefVector refVecJetColl ;
    l1extra::L1MuonParticleRefVector refVecMuonColl ;
    l1extra::L1EtMissParticleRefVector refVecEtMiss ;
    l1extra::L1HFRingsRefVector refVecHFRings ;

    l1extra::L1EmParticleVectorRef vecRefEmColl ;
    l1extra::L1JetParticleVectorRef vecRefJetColl ;
    l1extra::L1MuonParticleVectorRef vecRefMuonColl ;
    l1extra::L1EtMissParticleVectorRef vecRefEtMissColl ;
    l1extra::L1HFRingsVectorRef vecRefHFRingsColl ;

    l1extra::L1EtMissParticleRefProd refProdEtMiss ;

    edm::reftobase::Holder<reco::Candidate, l1extra::L1EmParticleRef> rtbe;
    edm::reftobase::Holder<reco::Candidate, l1extra::L1MuonParticleRef> rtbm;
    edm::reftobase::Holder<reco::Candidate, l1extra::L1JetParticleRef> rtbj;
    edm::reftobase::Holder<reco::Candidate, l1extra::L1EtMissParticleRef> rtbm1;
    edm::reftobase::Holder<reco::Candidate, l1extra::L1EtMissParticleRefProd> rtbm2;

    std::vector<l1extra::L1ParticleMap::L1ObjectType> dummy1 ;

    L1DataEmulRecord der;
    edm::Wrapper<L1DataEmulRecord> w_der;

    L1TriggerError l1tErr;
    L1TriggerErrorCollection l1tErrColl;
    edm::Wrapper<L1TriggerErrorCollection> w_l1terr;
  };
}
