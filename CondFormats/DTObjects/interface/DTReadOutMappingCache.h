#ifndef CondFormats_DTObjects_DTReadOutMappingCache_h
#define CondFormats_DTObjects_DTReadOutMappingCache_h

#include "CondFormats/DTObjects/interface/DTBufferTree.h"

#include <memory>
#include <vector>

template <class Key, class Content> class DTBufferTree;
class DTBufferTreeUniquePtr;

class DTReadOutMappingCache {

public:

  DTBufferTree<int,int> mType;
  DTBufferTree<int,int> rgBuf;
  DTBufferTree<int,int> rgROB;
  DTBufferTree<int,int> rgROS;
  DTBufferTree<int,int> rgDDU;
  DTBufferTree<int,int> grBuf;

  DTBufferTree<int, std::unique_ptr<std::vector<int> > > grROB;
  DTBufferTree<int, std::unique_ptr<std::vector<int> > > grROS;
  DTBufferTree<int, std::unique_ptr<std::vector<int> > > grDDU;
};
#endif
