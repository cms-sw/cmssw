#ifndef RECOCALOTOOLS_METACOLLECTIONS_CALORECHITMETACOLLECTIONV_H
#define RECOCALOTOOLS_METACOLLECTIONS_CALORECHITMETACOLLECTIONV_H 1

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include <iterator>



/** \class CaloRecHitMetaCollectionV
  *  
  * Virtual base class for a "meta collection" which references 
  * CaloRecHit-derived objects in their base collections.
  *
  * $Date: 2007/08/07 14:55:00 $
  * $Revision: 1.2 $
  * \author J. Mans - Minnesota
  */
class CaloRecHitMetaCollectionV {
public:
  class Iterator {
  public:
    typedef std::random_access_iterator_tag iterator_category;
    typedef const CaloRecHit& value_type;
    typedef int difference_type;
    typedef const CaloRecHit& reference;
    typedef const CaloRecHit* pointer;
    typedef int offset_type;

    Iterator() : collection_(0), offset_(0) { }
    Iterator(const Iterator& it) : collection_(it.collection_), offset_(it.offset_) { }
    Iterator(const CaloRecHitMetaCollectionV* col, offset_type pos) : collection_(col), offset_(pos) { }
    Iterator& operator=(const Iterator& it) { collection_=it.collection_; offset_=it.offset_; return (*this); }

    /// dereference operator
    reference operator*() const;
    /// pointer operator
    pointer operator->() const;

    /// comparison operator
    bool operator==(const Iterator& it) const;
    /// comparison operator
    bool operator!=(const Iterator& it) const;

    /// Advance the iterator
    Iterator& operator++();
    /// Advance the iterator
    Iterator operator++(int);
    /// Reverse-advance the iterator
    Iterator& operator--();
    /// Reverse-advance the iterator
    Iterator operator--(int);
    
    // Random-access iterator requirements
    reference operator[](const difference_type n) const;
    Iterator& operator+=(const difference_type n);
    Iterator operator+(const difference_type n) const;
    Iterator& operator-=(const difference_type n);
    Iterator operator-(const difference_type n) const;
    bool operator<(const Iterator& i) const; 

  private:
    const CaloRecHitMetaCollectionV* collection_;
    offset_type offset_;
  };
//
// add virtual descructor
//
  virtual ~CaloRecHitMetaCollectionV() {}
  typedef Iterator const_iterator;

  /// find by id (default version is very slow unsorted find)
  virtual const_iterator find(const DetId& id) const;

  /// get the starting iterator
  const_iterator begin() const { return const_iterator(this,0); }
  /// get the ending iterator
  const_iterator end() const { return const_iterator(this,(const_iterator::offset_type)(size_)); }
  /// get the size of the collection
  unsigned int size() const { return size_; }

  /// get an item by index
  virtual const CaloRecHit* at(const_iterator::offset_type i) const = 0;
  
protected:
  CaloRecHitMetaCollectionV();
  unsigned int size_; // must be updated by derived classes 
};

#endif
