#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "PhysicsTools/PatAlgos/interface/PATAuxiliaryProd.h"

namespace PhysicsTools_PatAlgos {
  
  struct dictionary {
    
    edm::Wrapper<edm::ValueMap<std::vector<uint8_t>> > dummy01;
    edm::Wrapper<edm::ValueMap<std::vector<unsigned char>> > dummy02;
    edm::Wrapper<edm::ValueMap<pat::SampleProd> > dummy03;
    edm::Wrapper<edm::ValueMap<pat::HcalDepthEnergyFractionProd> > dummy04;

    vector<pat::HcalDepthEnergyFractionProd> dummy05;
    vector<pat::SampleProd> dummy06;

    edm::ValueMap<pat::HcalDepthEnergyFractionProd> dummy07;
    edm::ValueMap<pat::SampleProd> dummy08;
  
    
  };

}
