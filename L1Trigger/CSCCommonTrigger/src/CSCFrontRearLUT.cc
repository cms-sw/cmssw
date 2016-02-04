#include <L1Trigger/CSCCommonTrigger/interface/CSCFrontRearLUT.h>

unsigned CSCFrontRearLUT::getFRBit(int sector, int subsector, int station, int cscid)
{
  unsigned dc=0, sector_type=0;
  unsigned fr_table[16][6]={{0,1,1,0,1,0},
			    {1,0,0,1,0,1},
			    {0,1,1,0,1,0},
			    {0,0,1,1,1,0},
			    {1,1,0,0,0,1},
			    {0,0,1,1,1,0},
			    {1,1,0,0,dc,dc},
			    {0,0,1,1,dc,dc},
			    {1,1,0,0,dc,dc},
			    {dc,dc,dc,dc,1,0},  // cscid 10-12 are me1a
			    {dc,dc,dc,dc,0,1},
			    {dc,dc,dc,dc,1,0},
			    {dc,dc,dc,dc,dc,dc},
			    {dc,dc,dc,dc,dc,dc},
			    {dc,dc,dc,dc,dc,dc},
			    {dc,dc,dc,dc,dc,dc}};

  switch(station)
    {
    case 1: sector_type = 4 + subsector;
      break;
    case 2: sector_type =  1 - (sector%2);
      break;
    case 3: sector_type = 3 - (sector%2);
      break;
    case 4: sector_type = 3 - (sector%2);
      break;
      //default:
      //std::cout << "+++ Error: unforeseen station " << stn << "in GetFRBit +++"; // replace with message logger or exception
    }
  return fr_table[cscid-1][sector_type];
}
