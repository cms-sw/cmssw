#ifndef RECOCALOTOOLS_METACOLLECTIONS_CALORECHITMETACOLLECTIONFAST_H
#define RECOCALOTOOLS_METACOLLECTIONS_CALORECHITMETACOLLECTIONFAST_H 1

#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionV.h"
#include <vector>

/** \class CaloRecHitMetaCollectionFast
  *  
  * Implementation of CaloRecHitMetaCollectionV which internally
  * stores a vector of const CaloRecHit pointers.  Fast for access and
  * relatively fast to build, but uses more memory than
  * CaloRecHitMetaCollectionCompact.
  *
  * Appropriate class for subcollections produced by selection algorithms.
  *
  * $Date: 2006/01/17 15:57:11 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class CaloRecHitMetaCollectionFast : public CaloRecHitMetaCollectionV {
public:
  typedef CaloRecHitMetaCollectionV::Iterator const_iterator;

  /// create an empty collection
  CaloRecHitMetaCollectionFast();
  /// copy constructor
  CaloRecHitMetaCollectionFast(const CaloRecHitMetaCollectionFast& c);
  /// destructor
  virtual ~CaloRecHitMetaCollectionFast() { }

  
  /// add an item to the collection
  void add(const CaloRecHit* hit);

  virtual const_iterator find(const DetId& id) const;
  virtual const CaloRecHit* at(const_iterator::offset_type i) const;

private:
  void sort() const;

  mutable std::vector<const CaloRecHit*> hits_;
  mutable bool dirty_;
};
#endif
