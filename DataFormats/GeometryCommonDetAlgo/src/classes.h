#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1DFloat.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    Measurement1D m;
    Measurement1DFloat f;
    edm::Wrapper<edm::ValueMap<Measurement1DFloat> > wvmf;
  };
}
