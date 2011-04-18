#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    TsosVectorCollection tsosesColl;
    edm::Wrapper<TsosVectorCollection> tsosesWrappedColl;
  };
}
