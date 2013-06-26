#include "CondFormats/DTObjects/interface/DTBufferTree.h"
#include <vector>

template<class Key, class Content>
Content DTBufferTree<Key,Content>::defaultContent;

class DTBufferInit {
 public:
// private:
  DTBufferInit() {
    DTBufferTree< int, int               >::setDefault( 0 );
    DTBufferTree< int, std::vector<int>* >::setDefault( 0 );
  }
};

DTBufferInit bufferInit;
