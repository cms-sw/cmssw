#ifndef EventFilter_CSCDigiToRaw_h
#define EventFilter_CSCDigiToRaw_h

/** \class CSCDigiToRaw
 *
 *  $Date: 2007/10/08 22:01:06 $
 *  $Revision: 1.5 $
 *  \author A. Tumanov - Rice
 */

#include <FWCore/Framework/interface/EDProducer.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/Common/interface/Handle.h>

class FEDRawDataCollection;
class CSCReadoutMappingFromFile;
class CSCEventData;
class CSCChamberMap;
class CSCDigiToRaw {
 public:
  /// Constructor
  CSCDigiToRaw();

  /// Destructor
  virtual ~CSCDigiToRaw();

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

};
#endif
