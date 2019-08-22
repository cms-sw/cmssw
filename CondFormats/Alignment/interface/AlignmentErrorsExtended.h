#ifndef CondFormats_Alignment_AlignmentErrorsExtended_H
#define CondFormats_Alignment_AlignmentErrorsExtended_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"

#include <vector>

class AlignmentErrorsExtended {
public:
  AlignmentErrorsExtended() {}
  virtual ~AlignmentErrorsExtended() {}
  /// Test of empty vector without having to look into internals:
  inline bool empty() const { return m_alignError.empty(); }
  /// Clear vector without having to look into internals:
  inline void clear() { m_alignError.clear(); }

  std::vector<AlignTransformErrorExtended> m_alignError;

  COND_SERIALIZABLE;
};
#endif  // AlignmentErrorsExtended_H
