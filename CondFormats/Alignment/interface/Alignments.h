#ifndef Alignments_H
#define Alignments_H

#include "CondFormats/Alignment/interface/AlignTransform.h"

#include<vector>

class Alignments {
public:
  Alignments(){}
  virtual ~Alignments(){}
  std::vector<AlignTransform> m_align;
};
#endif // Alignments_H
