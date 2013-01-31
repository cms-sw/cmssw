#ifndef CondFormats_Alignment_AlignmentErrors_H
#define CondFormats_Alignment_AlignmentErrors_H

#include "CondFormats/Alignment/interface/AlignTransformError.h"

#include<vector>

class AlignmentErrors {
public:
  AlignmentErrors(){}
  virtual ~AlignmentErrors(){}
  /// Test of empty vector without having to look into internals:
  inline bool empty() const { return m_alignError.empty();}
  /// Clear vector without having to look into internals:
  inline void clear() {m_alignError.clear();}

  std::vector<AlignTransformError> m_alignError;
};
#endif // AlignmentErrors_H
