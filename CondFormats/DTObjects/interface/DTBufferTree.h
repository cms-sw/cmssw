#ifndef DTBufferTree_H
#define DTBufferTree_H
/** \class DTBufferTree
 *
 *  Description:
 *
 *
 *  $Date: 2007/12/07 15:00:45 $
 *  $Revision: 1.3 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//---------------
// C++ Headers --
//---------------
#include <map>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class Key, class Content>
class DTBufferTree {

public:

  typedef typename std::vector<Key>               CompositeKey;
  typedef typename std::vector<Key>::const_iterator ElementKey;

  /** Constructor
   */
  DTBufferTree();

  /** Destructor
   */
  virtual ~DTBufferTree();

  /** Operations
   */
  /// 
  int insert( ElementKey fKey, ElementKey lKey, const Content& cont );
  int insert(  const Key& k,                    const Content& cont );
  int find(   ElementKey fKey, ElementKey lKey,       Content& cont );
  int find(    const Key& k,                          Content& cont );
  std::vector<Content> contList();
  static void setDefault( const Content& def );

 private:

  typedef DTBufferTree<Key,Content> map_node;
  typedef typename std::map<Key,DTBufferTree<Key,Content>*>           map_cont;
  typedef typename std::map<Key,DTBufferTree<Key,Content>*>::iterator map_iter;

  Content bufferContent;
  map_cont bufferMap;

  static Content defaultContent;

  void treeCont( std::vector<Content>& contentList );
  void leafCont( std::vector<Content>& contentList );

};

#include "CondFormats/DTObjects/interface/DTBufferTree.icc"

#endif // DTBufferTree_H

