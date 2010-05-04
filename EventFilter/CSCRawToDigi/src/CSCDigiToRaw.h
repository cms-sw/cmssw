#ifndef EventFilter_CSCDigiToRaw_h
#define EventFilter_CSCDigiToRaw_h

/** \class CSCDigiToRaw
 *
 *  $Date: 2010/04/23 23:03:04 $
 *  $Revision: 1.10 $
 *  \author A. Tumanov - Rice
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"

class FEDRawDataCollection;
class CSCReadoutMappingFromFile;
class CSCChamberMap;

class CSCDigiToRaw {
public:
  /// Constructor
  explicit CSCDigiToRaw(const edm::ParameterSet & pset);

  /// Take a vector of digis and fill the FEDRawDataCollection
  void createFedBuffers(const CSCStripDigiCollection& stripDigis,
			const CSCWireDigiCollection& wireDigis, 
                        const CSCComparatorDigiCollection& comparatorDigis,
                        const CSCALCTDigiCollection& alctDigis,
                        const CSCCLCTDigiCollection& clctDigis,
                        const CSCCLCTPreTriggerCollection& preTriggers,
                        const CSCCorrelatedLCTDigiCollection& correlatedLCTDigis,
			FEDRawDataCollection& fed_buffers,
		        const CSCChamberMap* theMapping, 
			edm::Event & e);

private:
  void beginEvent(const CSCChamberMap* electronicsMap);

  // specialized because it reverses strip direction
  void add(const CSCStripDigiCollection& stripDigis, 
           const CSCCLCTPreTriggerCollection& preTriggers);
  void add(const CSCWireDigiCollection& wireDigis);
  // may require CLCTs to read out comparators.  Doesn't add CLCTs.
  void add(const CSCComparatorDigiCollection & comparatorDigis,
           const CSCCLCTDigiCollection & clctDigis);
  void add(const CSCALCTDigiCollection & alctDigis);
  void add(const CSCCLCTDigiCollection & clctDigis);
  void add(const CSCCorrelatedLCTDigiCollection & corrLCTDigis);

  /// pick out the correct data object for this chamber
  CSCEventData & findEventData(const CSCDetId & cscDetId);

  /// takes layer ID, converts to chamber ID, switching ME1A to ME11
  CSCDetId chamberID(const CSCDetId & cscDetId) const;

  std::map<CSCDetId, CSCEventData> theChamberDataMap;
  const CSCChamberMap* theElectronicsMap;
  // used to zero-suppress strips
  bool requirePreTrigger_;
  bool requireCLCTForComparators_;
};




#endif
