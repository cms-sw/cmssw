#ifndef CSCCrateMap_h
#define CSCCrateMap_h

#include <CondFormats/CSCObjects/interface/CSCMapItem.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <map>

class CSCCrateMap{
 public:
  CSCCrateMap();
  ~CSCCrateMap();  

  /// Accessor for item according to a key
  const CSCMapItem::MapItem& item( int key ) const;

  /// Build DetId from hardware labels of vme crate, dmb
  /// Need cfeb to split ME11 into ME1a and ME1b.
  /// May need layer # 1-6 (set 0 for chamber, as default arg.)
  CSCDetId detId( int vme, int dmb, int cfeb, int layer = 0 ) const;

  typedef std::map< int,CSCMapItem::MapItem > CSCMap;
  CSCMap crate_map;
};

#endif
