/* -*- C++ -*- */
#ifndef HcalMapping_h_included
#define HcalMapping_h_included 1

#include <vector>
#include <map>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"


/** \class HcalMapping
 *  Map between electronics id and logical ids for use in Hcal Unpacking
 *
 *  Memory usage for electronics --> logical ~= 64kB (precision)
 *  Memory usage for logical --> electronics (optional) ~= 300kB 
 *  $Date: 2005/10/10 14:28:17 $
 *  $Revision: 1.3 $
 *  \author J. Mans - Minnesota
 */
class HcalMapping {
public:
  /** \brief Constructor 
      \param maintainL2E Flag to indicate if the
      Logical To Electrical map should be maintained.  This map
      requires significant memory and is generally not needed for most
      reconstruction operations.  It is needed for simulation.
  */
  HcalMapping(bool maintainL2E=false);
  /// clear the map
  void clear();
  
  /// Insert a mapping between an electronics id and a logical cell id
  void setMap(const HcalElectronicsId&, const HcalDetId&);
  /// Insert a mapping between an electronics id and a trigger tower logical cell id
  void setTriggerMap(const HcalElectronicsId&, const HcalTrigTowerDetId&);
  
  /// get an iterator over the logical cell ids
  std::vector<HcalDetId>::const_iterator detid_begin() const;
  /// get the ending iterator over the logical cell ids
  std::vector<HcalDetId>::const_iterator detid_end() const;
  
  /// get an iterator over the trigger logical cell ids
  std::vector<HcalTrigTowerDetId>::const_iterator trigger_detid_begin() const;
  /// get the ending iterator over the trigger logical cell ids
  std::vector<HcalTrigTowerDetId>::const_iterator trigger_detid_end() const;
  
  /** \brief lookup the logical detid associated with the given electronics id
      \return Null item if no such mapping
  */
  const HcalDetId& lookup(const HcalElectronicsId&) const;
  /** \brief lookup the electronics detid associated with the given logical id
      \return Null item if no such mapping
  */
  const HcalElectronicsId& lookup(const HcalDetId&) const;
  /** \brief lookup the trigger logical detid associated with the given electronics id
      \return Null item if no such mapping
  */
  const HcalTrigTowerDetId& lookupTrigger(const HcalElectronicsId&) const;
  /** \brief lookup the electronics detid associated with the given trigger logical id
      \return Null item if no such mapping
  */
  const HcalElectronicsId& lookupTrigger(const HcalTrigTowerDetId&) const;
  
  /** \brief Test if this subdetector is present in this dccid */
  bool subdetectorPresent(HcalSubdetector det, int dccid) const;

private:
  // flag to indicate if the logical to electrical map must be maintained
  bool maintainL2E_;
  // electronics to logical (usual use, must be fast)
  std::vector<HcalDetId> elecToLogical_;
  // electronics to trigger tower ...
  std::vector<HcalTrigTowerDetId> elecToTrigTower_;
  // logical to electronics (simulation only)
  std::map<HcalDetId,HcalElectronicsId> logicalToElec_;
  // logical to electronics (simulation only)
  std::map<HcalTrigTowerDetId,HcalElectronicsId> trigTowerToElec_;

  struct SubdetectorId {
    int hbheEntries;
    int hoEntries;
    int hfEntries;
  };
  SubdetectorId dccIds_[HcalElectronicsId::maxDCCId+1];
};

#endif // HcalMapping_h_included
