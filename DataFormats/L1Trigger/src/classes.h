#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtTotalPhys.h"
#include "DataFormats/L1Trigger/interface/L1EtHadPhys.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {

     l1extra::L1EmParticleCollection emCand ;
     l1extra::L1JetParticleCollection jetCand ;
     l1extra::L1MuonParticleCollection muonCand ;
     l1extra::L1EtMissParticle etMiss ;
     l1extra::L1EtTotalPhys etTot ;
     l1extra::L1EtHadPhys etHad ;

     edm::Wrapper<l1extra::L1EmParticleCollection> w_emCand;
     edm::Wrapper<l1extra::L1JetParticleCollection> w_jetCand;
     edm::Wrapper<l1extra::L1MuonParticleCollection> w_muonCand;
     edm::Wrapper<l1extra::L1EtMissParticle> w_etMiss;
     edm::Wrapper<l1extra::L1EtTotalPhys> w_etTot;
     edm::Wrapper<l1extra::L1EtHadPhys> w_etHad;
  }
}
