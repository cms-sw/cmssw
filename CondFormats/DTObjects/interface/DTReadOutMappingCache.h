#include "FWCore/Utilities/interface/ConstRespectingPtr.h"

template <class Key, class Content> class DTBufferTree;
class DTBufferTreeUniquePtr;

class DTReadOutMappingCache {

public:

  edm::ConstRespectingPtr<DTBufferTree<int,int> > mType;
  edm::ConstRespectingPtr<DTBufferTree<int,int> > rgBuf;
  edm::ConstRespectingPtr<DTBufferTree<int,int> > rgROB;
  edm::ConstRespectingPtr<DTBufferTree<int,int> > rgROS;
  edm::ConstRespectingPtr<DTBufferTree<int,int> > rgDDU;
  edm::ConstRespectingPtr<DTBufferTree<int,int> > grBuf;

  edm::ConstRespectingPtr<DTBufferTreeUniquePtr> grROB;
  edm::ConstRespectingPtr<DTBufferTreeUniquePtr> grROS;
  edm::ConstRespectingPtr<DTBufferTreeUniquePtr> grDDU;
};
