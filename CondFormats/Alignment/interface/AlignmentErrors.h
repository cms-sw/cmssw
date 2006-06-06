#ifndef AlignmentErrors_H
#define AlignmentErrors_H

#include "CondFormats/Alignment/interface/AlignTransformError.h"

#include<vector>

class AlignmentErrors {
public:
  AlignmentErrors(){}
  virtual ~AlignmentErrors(){}
  std::vector<AlignTransformError> m_alignError;
};
#endif // AlignmentErrors_H
