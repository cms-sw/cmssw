#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include <sstream>

std::string ChamberLocationSpec::print( int depth ) const
{
  std::ostringstream str;
  std::string subsVal[5]={"--","-","0","+","++"}; // -2,-1,0,1,2
  std::string febZVal[3]={"-z","0","+z"};         // -1,null,1
  std::string febZRVal[3]={"N/A","IN","OUT"};     // 0,1,2
  std::string boeVal[2]={"Barrel","Endcap"};      // 1,2
  if (depth >=0) {
    str  << " ChamberLocationSpec: " << std::endl
         << " --->DiskOrWheel: " << diskOrWheel
         << " Layer: " << layer
         << " Sector: " << sector
         << " Subsector: " << subsVal[subsector+2]
         << " ChamberLocationName: " << chamberLocationName()
         << " FebZOrnt: " << febZVal[febZOrnt+1]
         << " FebZRadOrnt: " << febZRVal[int(febZRadOrnt)]
         << " BarrelOrEndcap: " << boeVal[barrelOrEndcap-1];
  }
  return str.str();
}

std::string ChamberLocationSpec::chamberLocationName() const {
  std::ostringstream ocln;
  std::string cln;
  if (barrelOrEndcap == 1) {
    std::string layerVal[6]={"RB1in","RB1out","RB2in","RB2out","RB3","RB4"};
    std::string prefix="W";
    if (diskOrWheel > 0) prefix="W+";
    if (subsector == 0) {
      ocln << prefix<<diskOrWheel<<"/"<<layerVal[layer-1]<<"/"<<sector;
    } else {
      std::string subsVal[5]={"--","-","0","+","++"};
      ocln << prefix<<diskOrWheel<<"/"<<layerVal[layer-1]<<"/"<<sector<<subsVal[subsector+2];
    }
  } else {
    std::string prefix="RE";
    if (diskOrWheel > 0) prefix="RE+";
    ocln << prefix<<diskOrWheel<<"/"<<layer<<"/"<<sector;
  }
  cln=ocln.str();
  return cln;
}

