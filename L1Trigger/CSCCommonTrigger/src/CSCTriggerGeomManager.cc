#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeomManager.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <iostream>

std::vector<CSCChamber*> 
CSCTriggerGeomManager::sectorOfChambersInStation(unsigned endcap, unsigned station, 
						 unsigned sector, unsigned subsector) const
{
  std::vector<CSCChamber*> result;
  int ring = 0, chamber = 0;
  CSCDetId id;

  for(int tcscid = CSCTriggerNumbering::minTriggerCscId(); 
      tcscid <= CSCTriggerNumbering::maxTriggerCscId(); ++tcscid)
    {
      ring = CSCTriggerNumbering::ringFromTriggerLabels(station,tcscid);
      chamber = CSCTriggerNumbering::chamberFromTriggerLabels(sector,subsector,station,tcscid);
      
      id = CSCDetId(endcap,station,ring,chamber,0);
      
      result.push_back(const_cast<CSCChamber*>(geom->chamber(id)));
    }

  return result;
}

CSCChamber*
CSCTriggerGeomManager::chamber(unsigned endcap, unsigned station, 
			       unsigned sector, unsigned subsector, 
			       unsigned tcscid) const
{
  CSCChamber* result = NULL;
  
  int ring = 0;
  int chamber = 0;

  ring = CSCTriggerNumbering::ringFromTriggerLabels(station,tcscid);
  chamber = CSCTriggerNumbering::chamberFromTriggerLabels(sector,subsector,station,tcscid);
  CSCDetId id(endcap,station,ring,chamber,0);
  
  result = const_cast<CSCChamber*>(geom->chamber(id));
  
  return result;
}
