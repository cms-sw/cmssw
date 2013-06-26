//-------------------------------------------------
//
/**  \class DTCache
 *
 *   Trigger Cache
 *   Used to store various trigger data
 *    
 *
 *
 *   $Date: 2006/07/19 10:32:51 $
 *   $Revision: 1.1 $
 *
 *   \author  C. Battilana
 *
 *   Modifications:
 */
//
//--------------------------------------------------
#ifndef DT_CACHE_H
#define DT_CACHE_H

#include <vector>

template<class T, class Coll=std::vector<T> > class DTCache {

  public:
  
  typedef T                                           my_type;
  typedef Coll                                        my_collection;
  typedef typename my_collection::iterator            iterator;
  typedef typename my_collection::const_iterator      const_iterator;
  
  public:
  
  //! Constructor
  DTCache(){}
  
  //! Destructor
  virtual ~DTCache(){}
  
  //! Get first cache element
  const_iterator begin() const { return _cache.begin();}
  
  //! Get last cache element
  const_iterator end() const {return _cache.end();}
  
  //! Get cache vector's size
  int size() const {return _cache.size();}

  //! Clear cache vector
  void clearCache() {_cache.clear();}

  //! Virtual reconstruct member
  virtual void reconstruct() {}

  protected:
  
  my_collection _cache;

};
#endif
