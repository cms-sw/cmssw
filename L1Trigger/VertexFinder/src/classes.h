#include "DataFormats/Common/interface/Wrapper.h"
#include "L1Trigger/VertexFinder/interface/InputData.h"

namespace {
  struct dictionary {
	l1tVertexFinder::InputData id;
    edm::Wrapper<l1tVertexFinder::InputData> wid;
  };
}