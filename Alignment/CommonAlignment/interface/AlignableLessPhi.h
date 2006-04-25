#ifndef Alignment_CommonAlignment_AlignableLessPhi_H
#define Alignment_CommonAlignment_AlignableLessPhi_H

#include "CommonReco/DetAlignment/interface/Alignable.h"

/** Yet another class for defining an order in phi.
 */

class AlignableLessPhi {
public:
  bool operator()( const Alignable & a, const Alignable & b) {
    return a.position().phi() < b.position().phi();
  }
};

#endif // AlignableLessPhi_H
