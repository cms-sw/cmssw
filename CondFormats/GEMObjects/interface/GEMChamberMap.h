#ifndef GEMChamberMap_h
#define GEMChamberMap_h
// based on CSCChamberMap
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/GEMObjects/interface/GEMMapItem.h"
#include <map>

class GEMDetId;

class GEMChamberMap{
 public:
  GEMChamberMap();
  ~GEMChamberMap();

  /// Accessor for item according to key
  const GEMMapItem::MapItem& item( int key ) const;

  /// Interface required use in digi-to-raw

  /// vme crate id for given DetId
  int crate(const GEMDetId&) const;

  /// dmb id for given DetId
  int dmb(const GEMDetId&) const;

  /// ddu id for given DetId
  int ddu(const GEMDetId&) const;

  /// slink id for given DetId
  int slink(const GEMDetId&) const;

  /// ddu slot for given DetId
  int dduSlot(const GEMDetId&) const;

  /// ddu input for given DetId
  int dduInput(const GEMDetId&) const;

  /// Data are public. @@Should be private?
  typedef std::map< int, GEMMapItem::MapItem > GEMMap;
  GEMMap ch_map;

 private:
  /**
   * Decimal-encoded index (as used inside db - the 'Igor' index)
   *
   * This is the decimal integer ie*100000 + is*10000 + ir*1000 + ic*10 + il <br>
   * (ie=1-2, is=1-2, ir=1, ic=1-36, il=1-2) <br>
   * But in this case il=0 labels entire chamber.
   */
  int dbIndex(const GEMDetId&) const;

 COND_SERIALIZABLE;
};

#endif
