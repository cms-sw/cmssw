#ifndef GEOMETRY_CALOTOPOLOGY_CALOTOWERCONSTITUENTSMAP_H
#define GEOMETRY_CALOTOPOLOGY_CALOTOWERCONSTITUENTSMAP_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include <vector>
#include <map>
#include <atomic>

class HcalTopology;
class CaloTowerTopology;

/** \class CaloTowerConstituentsMap
  *  
  * \author J. Mans - Minnesota
  */
class CaloTowerConstituentsMap {
public:
  CaloTowerConstituentsMap() = delete;
  ~CaloTowerConstituentsMap();
  CaloTowerConstituentsMap(const HcalTopology * hcaltopo, const CaloTowerTopology * cttopo);

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
  const HcalTopology * m_hcaltopo;
  const CaloTowerTopology * m_cttopo;
    
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
  mutable std::atomic<std::multimap<CaloTowerDetId,DetId>*> m_reverseItems;
};

#endif
