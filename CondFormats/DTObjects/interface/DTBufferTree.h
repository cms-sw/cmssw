#ifndef DTBufferTree_H
#define DTBufferTree_H
/** \class DTBufferTree
 *
 *  Description:
 *
 *
 *  $Date: 2004/08/04 12:00:00 $
 *  $Revision: 1.1 $
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
  void   insert( ElementKey fKey, ElementKey lKey,
                 const Content& cont );
  Content& find( ElementKey fKey, ElementKey lKey );

 private:

  typedef DTBufferTree<Key,Content> map_node;
  typedef typename std::map<Key,DTBufferTree<Key,Content>*>           map_cont;
  typedef typename std::map<Key,DTBufferTree<Key,Content>*>::iterator map_iter;

  Content bufferContent;
  map_cont bufferMap;

  static Content defaultContent;

};

#include "CondFormats/DTObjects/interface/DTBufferTree.icc"

#endif // DTBufferTree_H

