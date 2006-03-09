#ifndef CSCCommonTrigger_CSCTriggerGeomManager_h
#define CSCCommonTrigger_CSCTriggerGeomManager_h


/** \class CSCTriggerGeomManager
Container for CSC geometry-related code.

\author Lindsey Gray    March 2006

--Port from ORCA L1MuCSCGeometryManager--
This class contains methods that provide access to the CSC Geometry 
using Trigger-Type labels. Based on nominal CSC geometry for now.
Updated to use CMSSW style pointers/interfaces.

*/

#include <FWCore/Framework/interface/ESHandle.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <vector>

class CSCGeometry;
class CSCTriggerGeomManager
{
 public:

  CSCTriggerGeomManager():geom(0) {}
  CSCTriggerGeomManager(const edm::ESHandle<CSCGeometry>& thegeom):geom(thegeom.product()) {}
  ~CSCTriggerGeomManager() {}

  /// Return a list of chambers in a given endcap/station/sector/subsector
  std::vector<Pointer2Chamber> sectorOfChambersInStation(unsigned endcap, unsigned station, 
						    unsigned sector, unsigned subsector) const;

  /// Return the CSCChamber for a corresponding endcap/station/sector/subsector/trigger cscid
  Pointer2Chamber chamber(unsigned endcap, unsigned station, unsigned sector,
			  unsigned subsector, unsigned tcscid) const;

 private:

  edm::ESHandle<CSCGeometry> geom;

};

#endif
