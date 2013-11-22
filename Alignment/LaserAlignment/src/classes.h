#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace Alignment_LaserAlignment {
  struct dictionary {
    TsosVectorCollection tsosesColl;
    edm::Wrapper<TsosVectorCollection> tsosesWrappedColl;
  };
}
