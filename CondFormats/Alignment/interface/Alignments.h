#ifndef Alignments_H
#define Alignments_H

#include "CondFormats/Alignment/interface/AlignTransform.h"

#include<vector>

class Alignments {
public:
  Alignments(){}
  virtual ~Alignments(){}
  /// Test of empty vector without having to look into internals:
  inline bool empty() const { return m_align.empty();}
  /// Clear vector without having to look into internals:
  inline void clear() {m_align.clear();}

  std::vector<AlignTransform> m_align;
};
#endif // Alignments_H
