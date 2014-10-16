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
#include "DataFormats/L1Trigger/interface/L1MuonParticleExtended.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleExtendedFwd.h" 
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

namespace {
  struct dictionary {

     l1extra::L1EmParticleCollection emColl ;
     l1extra::L1JetParticleCollection jetColl ;
     l1extra::L1MuonParticleCollection muonColl ;
     l1extra::L1MuonParticleExtendedCollection muonCollE ;
     l1extra::L1EtMissParticle etMiss ;
     l1extra::L1EtMissParticleCollection etMissColl ;
     l1extra::L1ParticleMapCollection mapColl ;
     l1extra::L1HFRingsCollection hfRingsColl ;

     edm::Wrapper<l1extra::L1EmParticleCollection> w_emColl;
     edm::Wrapper<l1extra::L1JetParticleCollection> w_jetColl;
     edm::Wrapper<l1extra::L1MuonParticleCollection> w_muonColl;
     edm::Wrapper<l1extra::L1MuonParticleExtendedCollection> w_muonCollE;
     edm::Wrapper<l1extra::L1EtMissParticle> w_etMiss;
     edm::Wrapper<l1extra::L1EtMissParticleCollection> w_etMissColl;
     edm::Wrapper<l1extra::L1ParticleMapCollection> w_mapColl;
     edm::Wrapper<l1extra::L1HFRingsCollection> w_hfRingsColl;

     l1extra::L1EmParticleRef refEm ;
     l1extra::L1JetParticleRef refJet ;
     l1extra::L1MuonParticleRef refMuon ;
     l1extra::L1MuonParticleExtendedRef refMuonE ;
     l1extra::L1EtMissParticleRef refEtMiss ;
     l1extra::L1HFRingsRef refHFRings ;

     l1extra::L1EmParticleRefVector refVecEmColl ;
     l1extra::L1JetParticleRefVector refVecJetColl ;
     l1extra::L1MuonParticleRefVector refVecMuonColl ;
     l1extra::L1MuonParticleExtendedRefVector refVecMuonCollE ;
     l1extra::L1EtMissParticleRefVector refVecEtMiss ;
     l1extra::L1HFRingsRefVector refVecHFRings ;

     l1extra::L1EmParticleVectorRef vecRefEmColl ;
     l1extra::L1JetParticleVectorRef vecRefJetColl ;
     l1extra::L1MuonParticleVectorRef vecRefMuonColl ;
     l1extra::L1MuonParticleExtendedVectorRef vecRefMuonCollE ;
     l1extra::L1EtMissParticleVectorRef vecRefEtMissColl ;
     l1extra::L1HFRingsVectorRef vecRefHFRingsColl ;

     l1extra::L1EtMissParticleRefProd refProdEtMiss ;

     edm::reftobase::Holder<reco::Candidate, l1extra::L1EmParticleRef> rtbe;
     edm::reftobase::Holder<reco::Candidate, l1extra::L1MuonParticleRef> rtbm;
     edm::reftobase::Holder<reco::Candidate, l1extra::L1MuonParticleExtendedRef> rtbmE;
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
