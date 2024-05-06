#include "DataFormats/HGCalReco/interface/TICLGraph.h"

bool operator==(ElementaryNode const& eN1, ElementaryNode const& eN2) {
  return (eN1.getId() == eN2.getId()) && (eN1.getNeighbours() == eN2.getNeighbours());
}

template <class T>
inline bool operator==(Node<T> const& n1, Node<T> const& n2) {
  return ((n1.getInternalStructure()) == (n2.getInternalStructure()));
}