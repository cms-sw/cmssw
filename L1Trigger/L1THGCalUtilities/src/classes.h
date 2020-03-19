#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

namespace L1Trigger_L1THGCalUtilities {
  struct dictionary {
    edm::RefProd<BXVector<l1t::HGCalTriggerCell> > dummyRefProd;
    edm::helpers::KeyVal<edm::RefProd<vector<CaloParticle> >, edm::RefProd<BXVector<l1t::HGCalTriggerCell> > >
        dummyKeyVal;
    edm::AssociationMap<edm::OneToMany<std::vector<CaloParticle>, BXVector<l1t::HGCalTriggerCell>, unsigned int> >
        dummyMap;
    edm::Wrapper<
        edm::AssociationMap<edm::OneToMany<std::vector<CaloParticle>, BXVector<l1t::HGCalTriggerCell>, unsigned int> > >
        dummyMapWrapper;
  };
}  // namespace L1Trigger_L1THGCalUtilities
