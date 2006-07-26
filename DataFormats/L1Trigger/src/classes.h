#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtTotalPhys.h"
#include "DataFormats/L1Trigger/interface/L1EtHadPhys.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {

     l1extra::L1EmParticleCollection emColl ;
     l1extra::L1JetParticleCollection jetColl ;
     l1extra::L1MuonParticleCollection muonColl ;
     l1extra::L1EtMissParticle etMiss ;
     l1extra::L1EtTotalPhys etTot ;
     l1extra::L1EtHadPhys etHad ;
     l1extra::L1ParticleMapCollection mapColl ;

     edm::Wrapper<l1extra::L1EmParticleCollection> w_emColl;
     edm::Wrapper<l1extra::L1JetParticleCollection> w_jetColl;
     edm::Wrapper<l1extra::L1MuonParticleCollection> w_muonColl;
     edm::Wrapper<l1extra::L1EtMissParticle> w_etMiss;
     edm::Wrapper<l1extra::L1EtTotalPhys> w_etTot;
     edm::Wrapper<l1extra::L1EtHadPhys> w_etHad;
     edm::Wrapper<l1extra::L1ParticleMapCollection> w_mapColl;
  }
}
