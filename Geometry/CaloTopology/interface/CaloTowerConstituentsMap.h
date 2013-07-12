#ifndef GEOMETRY_CALOTOPOLOGY_CALOTOWERCONSTITUENTSMAP_H
#define GEOMETRY_CALOTOPOLOGY_CALOTOWERCONSTITUENTSMAP_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include <vector>
#include <map>

class HcalTopology;

/** \class CaloTowerConstituentsMap
  *  
  * $Date: 2006/09/06 21:00:11 $
  * $Revision: 1.3 $
  * \author J. Mans - Minnesota
  */
class CaloTowerConstituentsMap {
public:
  CaloTowerConstituentsMap(const HcalTopology * topology);

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
  /// add standard (hardcoded) HO items?
  void useStandardHO(bool use=true);
  /// add standard (hardcoded) HF items?
  void useStandardHF(bool use=true);
  /// add standard (hardcoded) EB items?
  void useStandardEB(bool use=true);

private:
  const HcalTopology * m_topology;
    
  bool standardHB_;
  bool standardHE_;
  bool standardHF_;
  bool standardHO_;
  bool standardEB_;

  struct MapItem {
    typedef DetId key_type;
    MapItem(const DetId& acell, const CaloTowerDetId& atower) : cell(acell),tower(atower) { }
    DetId cell;
    CaloTowerDetId tower;
    inline DetId id() const { return cell; }
  };

  edm::SortedCollection<MapItem> m_items;
  mutable std::multimap<CaloTowerDetId,DetId> m_reverseItems;
};

#endif
