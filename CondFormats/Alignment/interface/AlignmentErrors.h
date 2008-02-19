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
  std::vector<AlignTransformError> m_alignError;
};
#endif // AlignmentErrors_H
