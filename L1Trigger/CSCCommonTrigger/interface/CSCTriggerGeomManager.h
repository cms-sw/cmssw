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

class CSCTriggerGeomManager
{
 public:

  CSCTriggerGeomManager():geom(0) {}
  CSCTriggerGeomManager(const CSCTriggerGeomManager& parent):geom(parent.geom) {}
  ~CSCTriggerGeomManager() {}

  void setGeometry(const edm::ESHandle<CSCGeometry>& thegeom) { geom = const_cast<CSCGeometry*>(thegeom.product()); }
  
  /// Return a list of chambers in a given endcap/station/sector/subsector
  std::vector<CSCChamber*> sectorOfChambersInStation(unsigned endcap, unsigned station, 
						     unsigned sector, unsigned subsector) const;

  /// Return the CSCChamber for a corresponding endcap/station/sector/subsector/trigger cscid
  CSCChamber* chamber(unsigned endcap, unsigned station, unsigned sector,
		      unsigned subsector, unsigned tcscid) const;

 private:

  CSCGeometry* geom;

};

#endif
