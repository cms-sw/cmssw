#include "CommonTools/ParticleFlow/plugins/TopProjector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

#include <iostream>

namespace
{
  // CV: implement operator<< for reco::CaloJets,
  //     needed by TopProjector to print debug output
  //    (code does not compile without operator<< implementation)
  std::ostream& operator<<(std::ostream& stream, const reco::CaloJet& caloJet) 
  { 
    stream << "CaloJet: Pt = " << caloJet.pt() << ", eta = " << caloJet.eta() << ", phi = " << caloJet.phi() << std::endl;
    return stream;
  } 
}

typedef TopProjector<reco::CaloJet, CaloTower, std::vector<reco::CaloJet>, CaloTowerCollection> TPCaloJetsOnCaloTowers;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TPCaloJetsOnCaloTowers);


