#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace PhysicsTools_PatAlgos {

  struct dictionary {
    
    edm::Wrapper<edm::ValueMap<std::vector<uint8_t>> > dummy01;
    edm::Wrapper<edm::ValueMap<std::vector<unsigned char>> > dummy02;
    //edm::Wrapper<edm::ValueMap<std::vector<SampleProd>> > dummy03;

  };

}
