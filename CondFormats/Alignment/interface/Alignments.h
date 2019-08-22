#ifndef Alignments_H
#define Alignments_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/Alignment/interface/AlignTransform.h"

#include <vector>

class Alignments {
public:
  Alignments() {}
  virtual ~Alignments() {}
  /// Test of empty vector without having to look into internals:
  inline bool empty() const { return m_align.empty(); }
  /// Clear vector without having to look into internals:
  inline void clear() { m_align.clear(); }

  std::vector<AlignTransform> m_align;

  COND_SERIALIZABLE;
};
#endif  // Alignments_H
