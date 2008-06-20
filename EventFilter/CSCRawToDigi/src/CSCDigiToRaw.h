#ifndef EventFilter_CSCDigiToRaw_h
#define EventFilter_CSCDigiToRaw_h

/** \class CSCDigiToRaw
 *
 *  $Date: 2008/06/11 02:12:57 $
 *  $Revision: 1.7 $
 *  \author A. Tumanov - Rice
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"

class FEDRawDataCollection;
class CSCReadoutMappingFromFile;
class CSCChamberMap;

class CSCDigiToRaw {
public:
  /// Constructor
  CSCDigiToRaw();

  void beginEvent(const CSCChamberMap* electronicsMap);

  // specialized because it reverses strip direction
  void add(const CSCStripDigiCollection& stripDigis);
  void add(const CSCWireDigiCollection& wireDigis);
  void add(const CSCComparatorDigiCollection & comparatorDigis);
  void add(const CSCALCTDigiCollection & alctDigis);
  void add(const CSCCLCTDigiCollection & clctDigis);
  void add(const CSCCorrelatedLCTDigiCollection & corrLCTDigis);

  /// Take a vector of digis and fill the FEDRawDataCollection
  void createFedBuffers(const CSCStripDigiCollection& stripDigis,
			const CSCWireDigiCollection& wireDigis, 
			FEDRawDataCollection& fed_buffers,
		        const CSCChamberMap* theMapping, 
			edm::Event & e);

  std::map <CSCDetId, CSCEventData> fillChamberDataMap(const CSCStripDigiCollection& stripDigis,
						       const CSCWireDigiCollection& wireDigis,
						       const CSCChamberMap* theMapping);

private:
  /// pick out the correct data object for this chamber
  CSCEventData & findEventData(const CSCDetId & cscDetId);

  std::map<CSCDetId, CSCEventData> theChamberDataMap;
  const CSCChamberMap* theElectronicsMap;
};




#endif
