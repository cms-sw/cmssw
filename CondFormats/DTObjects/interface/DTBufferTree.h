#ifndef DTBufferTree_H
#define DTBufferTree_H
/** \class DTBufferTree
 *
 *  Description:
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

#include <map>
#include <memory>
#include <vector>

// This trait template defines the type of the output argument
// in the find function.  Usually the output type is just the
// content type, but in the special case where the content is
// a unique_ptr the the output type is a bare pointer. The
// functions in the trait template make that output type work
// in both cases.
template <typename T>
class DTBufferTreeTrait {
public:
  typedef T outputTypeOfConstFind;
  typedef T outputTypeOfNonConstFind;

  static T getOutputValue(T const& t) { return t; }
  static T getDefault() { return 0; }
};

template <typename T>
class DTBufferTreeTrait<std::unique_ptr<T> > {
public:
  typedef T const* outputTypeOfConstFind;
  typedef T* outputTypeOfNonConstFind;

  static T* getOutputValue(std::unique_ptr<T> const& t) { return t.get(); }
  static T* getDefault() { return nullptr; }
};

template <class Key, class Content>
class DTBufferTree {

public:

  typedef typename std::vector<Key>::const_iterator ElementKey;

  DTBufferTree();
  virtual ~DTBufferTree();

  void clear();

  int insert(ElementKey fKey, ElementKey lKey, Content cont );
  int insert(const Key& k,                     Content cont );
  int find(ElementKey fKey, ElementKey lKey, typename DTBufferTreeTrait<Content>::outputTypeOfConstFind& cont ) const;
  int find(const Key& k,                     typename DTBufferTreeTrait<Content>::outputTypeOfConstFind& cont ) const;
  int find(ElementKey fKey, ElementKey lKey, typename DTBufferTreeTrait<Content>::outputTypeOfNonConstFind& cont );
  int find(const Key& k,                     typename DTBufferTreeTrait<Content>::outputTypeOfNonConstFind& cont );

private:

  DTBufferTree(DTBufferTree const&) = delete;
  DTBufferTree& operator=(DTBufferTree const&) = delete;

  typedef DTBufferTree<Key,Content> map_node;
  typedef typename std::map<Key,DTBufferTree<Key,Content>*> map_cont;
  typedef typename std::map<Key,DTBufferTree<Key,Content>*>::const_iterator map_iter;

  Content bufferContent;
  map_cont bufferMap;
};

#include "CondFormats/DTObjects/interface/DTBufferTree.icc"

// This class is defined because it is easier to forward declare
// it than the template. Then unique_ptr is not exposed. When
// we stop using GCCXML (which does not recognize C++11's unique_ptr)
// this will not be as important, although it will still keep
// #include <memory> out of the header file of the class using
// DTBufferTree.
class DTBufferTreeUniquePtr : public DTBufferTree<int, std::unique_ptr<std::vector<int> > > {
};

#endif // DTBufferTree_H
