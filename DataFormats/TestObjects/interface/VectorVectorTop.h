#ifndef DataFormats_TestObjects_VectorVectorTop_h
#define DataFormats_TestObjects_VectorVectorTop_h

/** \class VectorVectorTop

\author W. David Dagenhart, created 21 July, 2023

*/

#include "DataFormats/TestObjects/interface/SchemaEvolutionTestObjects.h"

#include <vector>

namespace edmtest {

  class VectorVectorMiddle {
  public:
    VectorVectorMiddle();
    std::vector<VectorVectorElement> middleVector_;
  };

  class VectorVectorTop {
  public:
    VectorVectorTop();
    std::vector<VectorVectorMiddle> outerVector_;
  };

  class VectorVectorMiddleNonSplit {
  public:
    VectorVectorMiddleNonSplit();
    std::vector<VectorVectorElementNonSplit> middleVector_;
  };

  class VectorVectorTopNonSplit {
  public:
    VectorVectorTopNonSplit();
    std::vector<VectorVectorMiddleNonSplit> outerVector_;
  };

}  // namespace edmtest
#endif
