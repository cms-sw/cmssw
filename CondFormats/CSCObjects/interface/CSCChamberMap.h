#ifndef CSCChamberMap_h
#define CSCChamberMap_h

#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
#include <map>

class CSCDetId;

class CSCChamberMap{
 public:
  CSCChamberMap();
  ~CSCChamberMap();

  /// Accessor for item according to key
  const CSCMapItem::MapItem& item( int key ) const;

  /// Interface required use in digi-to-raw

  /// vme crate id for given DetId
  int crate(const CSCDetId&) const;

  /// dmb id for given DetId
  int dmb(const CSCDetId&) const;

  /// ddu id for given DetId
  int ddu(const CSCDetId&) const;

  /// slink id for given DetId
  int slink(const CSCDetId&) const;

  /// ddu slot for given DetId
  int dduSlot(const CSCDetId&) const;

  /// ddu input for given DetId
  int dduInput(const CSCDetId&) const;

  /// Data are public. @@Should be private?
  typedef std::map< int, CSCMapItem::MapItem > CSCMap;
  CSCMap ch_map;

 private:
  /**
   * Decimal-encoded index (as used inside db - the 'Igor' index)
   *
   * This is the decimal integer ie*100000 + is*10000 + ir*1000 + ic*10 + il <br>
   * (ie=1-2, is=1-4, ir=1-4, ic=1-36, il=1-6) <br>
   * But in this case il=0 labels entire chamber.
   */
  int dbIndex(const CSCDetId&) const;
};

#endif
