#ifndef EventFilter_GEMRawToDigi_h
#define EventFilter_GEMRawToDigi_h

/** \class GEMRawToDigi
 *  \based on CSCDigiToRaw
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "EventFilter/GEMRawToDigi/interface/GEMAMC13EventFormat.h"
#include "EventFilter/GEMRawToDigi/interface/GEMDataAMCformat.h"
#include "EventFilter/GEMRawToDigi/interface/GEMslotContents.h"
#include "EventFilter/GEMRawToDigi/interface/GEMDataChecker.h"

#include "CondFormats/GEMObjects/interface/GEMROmap.h"
#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/DataRecord/interface/GEMEMapRcd.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMEventData.h"

class FEDRawDataCollection;
class GEMReadoutMappingFromFile;
class GEMChamberMap;

class GEMRawToDigi {
 public:
  /// Constructor
  explicit GEMRawToDigi(const edm::ParameterSet & pset){};
  
  /// Take a vector of digis and fill the FEDRawDataCollection
  void readFedBuffers(const GEMDigiCollection& gemDigi,
		      const GEMPadDigiCollection& gemPadDigi,
		      const GEMPadDigiClusterCollection& gemPadDigiCluster,
		      const GEMCoPadDigiCollection& gemCoPadDigi,			
		      FEDRawDataCollection& fed_buffers,
		      const GEMChamberMap* theMapping, 
		      edm::Event & e);

 private:
  void beginEvent(const GEMChamberMap* map){theChamberMap = map;};

  // specialized because it reverses strip direction
  void read(const GEMDigiCollection& digis);
  void read(const GEMPadDigiCollection& digis);
  void read(const GEMPadDigiClusterCollection& digis);
  void read(const GEMCoPadDigiCollection& digis);
  /// pick out the correct data object for this chamber
  GEMEventData & findEventData(const GEMDetId & gemDetId);

  std::map<GEMDetId, GEMEventData> theChamberDataMap;
  const GEMChamberMap* theChamberMap;
};
#endif
