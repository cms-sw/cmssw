
#include "DataFormats/TestObjects/interface/SchemaEvolutionTestObjects.h"

namespace edmtest {
  VectorVectorElement::VectorVectorElement() : a_(0), b_(0) {}
  VectorVectorElement::VectorVectorElement(int a,
                                           int b,
                                           SchemaEvolutionChangeOrder const& changeOrder,
                                           SchemaEvolutionAddMember const& addMember,
                                           SchemaEvolutionRemoveMember const& removeMember,
                                           SchemaEvolutionMoveToBase const& moveToBase,
                                           SchemaEvolutionChangeType const& changeType,
                                           SchemaEvolutionAddBase const& addBase,
                                           SchemaEvolutionPointerToMember const& pointerToMember,
                                           SchemaEvolutionPointerToUniquePtr const& pointerToUniquePtr,
                                           SchemaEvolutionCArrayToStdArray const& cArrayToStdArray,
                                           // SchemaEvolutionCArrayToStdVector const& cArrayToStdVector,
                                           SchemaEvolutionVectorToList const& vectorToList,
                                           SchemaEvolutionMapToUnorderedMap const& mapToUnorderedMap)
      : a_(a),
        b_(b),
        changeOrder_(changeOrder),
        addMember_(addMember),
        removeMember_(removeMember),
        moveToBase_(moveToBase),
        changeType_(changeType),
        addBase_(addBase),
        pointerToMember_(pointerToMember),
        pointerToUniquePtr_(pointerToUniquePtr),
        cArrayToStdArray_(cArrayToStdArray),
        // cArrayToStdVector_(cArrayToStdVector),
        vectorToList_(vectorToList),
        mapToUnorderedMap_(mapToUnorderedMap) {}

#if defined DataFormats_TestObjects_USE_OLD
  VectorVectorElementNonSplit::VectorVectorElementNonSplit() : a_(0) {}
  VectorVectorElementNonSplit::VectorVectorElementNonSplit(int a, int) : a_(a) {}
#else
  VectorVectorElementNonSplit::VectorVectorElementNonSplit() : a_(0), b_(0) {}
  VectorVectorElementNonSplit::VectorVectorElementNonSplit(int a, int b) : a_(a), b_(b) {}
#endif

}  // namespace edmtest
