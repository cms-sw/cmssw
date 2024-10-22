#ifndef EventFilter_CSCRawToDigi_CSCDigiToRaw_h
#define EventFilter_CSCRawToDigi_CSCDigiToRaw_h

/** \class CSCDigiToRaw
 *
 *  \author A. Tumanov - Rice
 */

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FEDRawDataCollection;
class CSCChamberMap;

class CSCDigiToRaw {
public:
  /// CSC Shower HMT bits types
  enum CSCShowerType { lctShower = 0, anodeShower = 1, cathodeShower = 2, anodeALCTShower = 3 };

  /// Constructor
  explicit CSCDigiToRaw(const edm::ParameterSet& pset);

  /// Take a vector of digis and fill the FEDRawDataCollection
  void createFedBuffers(const CSCStripDigiCollection& stripDigis,
                        const CSCWireDigiCollection& wireDigis,
                        const CSCComparatorDigiCollection& comparatorDigis,
                        const CSCALCTDigiCollection& alctDigis,
                        const CSCCLCTDigiCollection& clctDigis,
                        const CSCCLCTPreTriggerCollection* preTriggers,
                        const CSCCLCTPreTriggerDigiCollection* preTriggerDigis,
                        const CSCCorrelatedLCTDigiCollection& correlatedLCTDigis,
                        const CSCShowerDigiCollection* showerDigis,
                        const CSCShowerDigiCollection* anodeShowerDigis,
                        const CSCShowerDigiCollection* cathodeShowerDigis,
                        const CSCShowerDigiCollection* anodeALCTShowerDigis,
                        const GEMPadDigiClusterCollection* padDigiClusters,
                        FEDRawDataCollection& fed_buffers,
                        const CSCChamberMap* theMapping,
                        const edm::EventID& eid) const;

private:
  struct FindEventDataInfo {
    FindEventDataInfo(const CSCChamberMap* map, uint16_t version) : theElectronicsMap{map}, formatVersion_{version} {}

    using ChamberDataMap = std::map<CSCDetId, CSCEventData>;
    ChamberDataMap theChamberDataMap;
    const CSCChamberMap* theElectronicsMap;
    const uint16_t formatVersion_;
  };

  // specialized because it reverses strip direction
  void add(const CSCStripDigiCollection& stripDigis,
           const CSCCLCTPreTriggerCollection* preTriggers,
           const CSCCLCTPreTriggerDigiCollection* preTriggerDigis,
           FindEventDataInfo&) const;
  void add(const CSCWireDigiCollection& wireDigis, const CSCALCTDigiCollection& alctDigis, FindEventDataInfo&) const;
  // may require CLCTs to read out comparators.  Doesn't add CLCTs.
  void add(const CSCComparatorDigiCollection& comparatorDigis,
           const CSCCLCTDigiCollection& clctDigis,
           FindEventDataInfo&) const;
  void add(const CSCALCTDigiCollection& alctDigis, FindEventDataInfo&) const;
  void add(const CSCCLCTDigiCollection& clctDigis, FindEventDataInfo&) const;
  void add(const CSCCorrelatedLCTDigiCollection& corrLCTDigis, FindEventDataInfo&) const;
  /// Run3 packing of CSCShower objects depending on shower HMT type
  void add(const CSCShowerDigiCollection& cscShowerDigis,
           FindEventDataInfo&,
           enum CSCShowerType shower = CSCShowerType::lctShower) const;
  /// Run3 adding GEM GE11 Pad Clusters trigger objects
  void add(const GEMPadDigiClusterCollection& gemPadClusters, FindEventDataInfo&) const;
  /// pick out the correct data object for this chamber
  CSCEventData& findEventData(const CSCDetId& cscDetId, FindEventDataInfo&) const;

  const int alctWindowMin_;
  const int alctWindowMax_;
  const int clctWindowMin_;
  const int clctWindowMax_;
  const int preTriggerWindowMin_;
  const int preTriggerWindowMax_;

  uint16_t formatVersion_;
  bool packEverything_;
  bool usePreTriggers_;
  bool packByCFEB_;
};

#endif
