/* -*- C++ -*- */
#ifndef HcalMapping_h_included
#define HcalMapping_h_included 1

#include <vector>
#include <map>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

namespace cms {

  namespace hcal {

    /** \class HcalMapping
     *  Map between electronics id and logical ids for use in Hcal Unpacking
     *
     *  Memory usage for electronics --> logical ~= 64kB (precision)
     *  Memory usage for logical --> electronics (optional) ~= 300kB 
     *  $Date: 2005/06/06 19:30:26 $
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
      void setMap(const cms::HcalElectronicsId&, const cms::HcalDetId&);
      /// Insert a mapping between an electronics id and a trigger tower logical cell id
      void setTriggerMap(const cms::HcalElectronicsId&, const cms::HcalTrigTowerDetId&);
      
      /// get an iterator over the logical cell ids
      std::vector<cms::HcalDetId>::const_iterator detid_begin() const;
      /// get the ending iterator over the logical cell ids
      std::vector<cms::HcalDetId>::const_iterator detid_end() const;
      
      /// get an iterator over the trigger logical cell ids
      std::vector<cms::HcalTrigTowerDetId>::const_iterator trigger_detid_begin() const;
      /// get the ending iterator over the trigger logical cell ids
      std::vector<cms::HcalTrigTowerDetId>::const_iterator trigger_detid_end() const;
      
      /** \brief lookup the logical detid associated with the given electronics id
	  \return Null item if no such mapping
      */
      const cms::HcalDetId& lookup(const cms::HcalElectronicsId&) const;
      /** \brief lookup the electronics detid associated with the given logical id
	  \return Null item if no such mapping
      */
      const cms::HcalElectronicsId& lookup(const cms::HcalDetId&) const;
      /** \brief lookup the trigger logical detid associated with the given electronics id
      \return Null item if no such mapping
      */
      const cms::HcalTrigTowerDetId& lookupTrigger(const cms::HcalElectronicsId&) const;
      /** \brief lookup the electronics detid associated with the given trigger logical id
	  \return Null item if no such mapping
      */
      const cms::HcalElectronicsId& lookupTrigger(const cms::HcalTrigTowerDetId&) const;
      
      /** \brief Get the majority detector type for the given dcc id.  
      \note HB and HE are both identified using "HcalBarrel".
      */
      cms::HcalSubdetector majorityDetector(int dccid) const;
    private:
      // flag to indicate if the logical to electrical map must be maintained
      bool maintainL2E_;
      // electronics to logical (usual use, must be fast)
      std::vector<cms::HcalDetId> elecToLogical_;
      // electronics to trigger tower ...
      std::vector<cms::HcalTrigTowerDetId> elecToTrigTower_;
  // logical to electronics (simulation only)
      std::map<cms::HcalDetId,cms::HcalElectronicsId> logicalToElec_;
      // logical to electronics (simulation only)
      std::map<cms::HcalTrigTowerDetId,cms::HcalElectronicsId> trigTowerToElec_;
      // majority-vote identity of each DCC as one subdetector (HB/HE considered one)
      struct SubdetectorId {
	int hbheEntries;
	int hoEntries;
	int hfEntries;
	cms::HcalSubdetector majorityId;
      };
      SubdetectorId dccIds_[cms::HcalElectronicsId::maxDCCId+1];
    };
  }
}

#endif // HcalMapping_h_included
