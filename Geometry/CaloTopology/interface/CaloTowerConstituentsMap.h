#ifndef GEOMETRY_CALOTOPOLOGY_CALOTOWERCONSTITUENTSMAP_H
#define GEOMETRY_CALOTOPOLOGY_CALOTOWERCONSTITUENTSMAP_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include <vector>

/** \class CaloTowerConstituentsMap
  *  
  * $Date: 2005/10/06 00:42:49 $
  * $Revision: 1.2 $
  * \author J. Mans - Minnesota
  */
class CaloTowerConstituentsMap {
public:
  CaloTowerConstituentsMap();

  /// Get the tower id for this det id (or null if not known)
  CaloTowerDetId towerOf(const DetId& id) const;

  /// Get the constituent detids for this tower id ( not yet implemented )
  std::vector<DetId> constituentsOf(const CaloTowerDetId& id) const;

  /// set the association between a DetId and a tower
  void assign(const DetId& cell, const CaloTowerDetId& tower);

  /// done adding to the association
  void sort();

  /// add standard (hardcoded) HB items?
  void useStandardHB(bool use=true);
  /// add standard (hardcoded) HE items?
  void useStandardHE(bool use=true);
  /// add standard (hardcoded) HF items?
  void useStandardHF(bool use=true);
  /// add standard (hardcoded) EB items?
  void useStandardEB(bool use=true);

private:
  bool standardHB_;
  bool standardHE_;
  bool standardHF_;
  bool standardEB_;

  struct MapItem {
    typedef DetId key_type;
    MapItem(const DetId& acell, const CaloTowerDetId& atower) : cell(acell),tower(atower) { }
    DetId cell;
    CaloTowerDetId tower;
    inline DetId id() const { return cell; }
  };

  edm::SortedCollection<MapItem> m_items;
};

#endif
