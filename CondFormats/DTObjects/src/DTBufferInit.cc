#include "CondFormats/DTObjects/interface/DTDataBuffer.h"
#include "CondFormats/DTObjects/interface/DTBufferTree.h"
#include <vector>

template<class Key, class Content>
std::map<std::string,DTBufferTree<Key,Content>*>
DTDataBuffer<Key,Content>::bufferMap;

template<class Key, class Content>
Content DTBufferTree<Key,Content>::defaultContent;

class DTBufferInit {
 public:
// private:
  DTBufferInit() {
//    DTDataBuffer< int, int >               db_int_int;
//    DTDataBuffer< int, std::vector<int>* > db_int_pvec_int;
    DTDataBuffer< int, int               >::dropBuffer( " " );
    DTDataBuffer< int, std::vector<int>* >::dropBuffer( " " );
    DTBufferTree< int, int               >::setDefault( 0 );
    DTBufferTree< int, std::vector<int>* >::setDefault( 0 );
/*
    DTBufferTree< int, int               > bt_int_int;
    DTBufferTree< int, std::vector<int>* > bt_int_pvec_int;
    int               x_int_int;
    std::vector<int>* x_int_pvec_int;
    int
    status = bt_int_int     .find( 0, x_int_int );
    status = bt_int_pvec_int.find( 0, x_int_pvec_int );
*/
  }
};

DTBufferInit bufferInit;
