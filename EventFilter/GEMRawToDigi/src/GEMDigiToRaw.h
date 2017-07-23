#ifndef EventFilter_GEMDigiToRaw_h
#define EventFilter_GEMDigiToRaw_h

/** \class GEMDigiToRaw
 *  \based on CSCDigiToRaw
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMEventData.h"

class FEDRawDataCollection;
class GEMReadoutMappingFromFile;
class GEMChamberMap;

class GEMDigiToRaw {
public:
  /// Constructor
  explicit GEMDigiToRaw(const edm::ParameterSet & pset){};
  
  /// Take a vector of digis and fill the FEDRawDataCollection
  void createFedBuffers(const GEMDigiCollection& gemDigi,
			const GEMPadDigiCollection& gemPadDigi,
			const GEMPadDigiClusterCollection& gemPadDigiCluster,
			const GEMCoPadDigiCollection& gemCoPadDigi,			
			FEDRawDataCollection& fed_buffers,
		        const GEMChamberMap* theMapping, 
			edm::Event & e);

private:
  void beginEvent(const GEMChamberMap* electronicsMap);

  // specialized because it reverses strip direction
  void add(const GEMDigiCollection& digis);
  void add(const GEMPadDigiCollection& digis);
  void add(const GEMPadDigiClusterCollection& digis);
  void add(const GEMCoPadDigiCollection& digis);
  /// pick out the correct data object for this chamber
  GEMEventData & findEventData(const GEMDetId & gemDetId);

  std::map<GEMDetId, GEMEventData> theChamberDataMap;
  const GEMChamberMap* theElectronicsMap;
};
#endif
