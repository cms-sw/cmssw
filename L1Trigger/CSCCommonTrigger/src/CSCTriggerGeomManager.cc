#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeomManager.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>


std::vector<const CSCChamber*> 
CSCTriggerGeomManager::sectorOfChambersInStation(unsigned endcap, unsigned station, 
						 unsigned sector, unsigned subsector) const
{
  std::vector<const CSCChamber*> result;
  int ring = 0, chamber = 0;
  CSCDetId id;

  for(int tcscid = CSCTriggerNumbering::minTriggerCscId(); 
      tcscid <= CSCTriggerNumbering::maxTriggerCscId(); ++tcscid)
    {
      try
	{
	  ring = CSCTriggerNumbering::ringFromTriggerLabels(station,tcscid);
	  chamber = CSCTriggerNumbering::chamberFromTriggerLabels(sector,subsector,station,tcscid);
	  
	  id = CSCDetId(endcap,station,ring,chamber,0);
	  
	  result.push_back(geom->chamber(id));
	}
      catch(...) {}
    }

  return result;
}

const CSCChamber*
CSCTriggerGeomManager::chamber(unsigned endcap, unsigned station, 
			       unsigned sector, unsigned subsector, 
			       unsigned tcscid) const
{
  CSCChamber* result = NULL;
  
  int ring = 0;
  int chamber = 0;

  try
    {
      ring = CSCTriggerNumbering::ringFromTriggerLabels(station,tcscid);
      chamber = CSCTriggerNumbering::chamberFromTriggerLabels(sector,subsector,station,tcscid);
      CSCDetId id(endcap,station,ring,chamber,0);

      result = const_cast<CSCChamber*>(geom->chamber(id));
    }
  catch(...) {}  
  
  return const_cast<const CSCChamber*>(result);
}
