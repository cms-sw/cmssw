#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>
#include <iostream>

CSCWireHit::CSCWireHit() :
  theDetId(),
  theWireHitPosition(),
  theWgroups(),
  theWireHitTmax(),
  isDeadWGAround()
{
theWgroupsHighBits.clear();
for(int i=0; i<(int)theWgroups.size(); i++)
   theWgroupsHighBits.push_back((theWgroups[i] >> 16) & 0x0000FFFF);
theWgroupsLowBits.clear();
for(int i=0; i<(int)theWgroups.size(); i++)
   theWgroupsLowBits.push_back(theWgroups[i] & 0x0000FFFF);
}

CSCWireHit::CSCWireHit( const CSCDetId& id, 
                        const float& wHitPos, 
                        ChannelContainer& wgroups, 
                        const int& tmax,
                        const bool& isNearDeadWG ) :
  theDetId( id ), 
  theWireHitPosition( wHitPos ),
  theWgroups( wgroups ),
  theWireHitTmax ( tmax ),
  isDeadWGAround( isNearDeadWG )
{
theWgroupsHighBits.clear();
for(int i=0; i<(int)theWgroups.size(); i++)
   theWgroupsHighBits.push_back((theWgroups[i] >> 16) & 0x0000FFFF);
theWgroupsLowBits.clear();
for(int i=0; i<(int)theWgroups.size(); i++)
   theWgroupsLowBits.push_back(theWgroups[i] & 0x0000FFFF);
}

CSCWireHit::~CSCWireHit() {}

/// Debug
void
CSCWireHit::print() const {
   std::cout << " CSCWireHit in CSC Detector: " << std::dec << cscDetId() << std::endl;
   std::cout << " wHitPos: " << wHitPos() << std::endl;
   std::cout << " BX + WireGroups combined: ";
   for (int i=0; i<(int)wgroupsBXandWire().size(); i++) {std::cout //std::dec << wgroups()[i] 
        << "HEX: " << std::hex << wgroupsBXandWire()[i] << std::hex << " ";
   }
   std::cout << std::endl;
   std::cout << " WireGroups: ";
   for (int i=0; i<(int)wgroups().size(); i++) {std::cout << std::dec << wgroups()[i] 
       << " (" << "HEX: " << std::hex << wgroups()[i] << ")" << " ";
   }
   std::cout << " BX#: ";
   for (int i=0; i<(int)wgroupsBX().size(); i++) {std::cout << std::dec << wgroupsBX()[i] 
       << " (" << "HEX: " << std::hex << wgroupsBX()[i] << ")" << " ";
   }
   
   std::cout << std::endl;
   std::cout << " TMAX: " << std::dec << tmax() << std::endl;
   std::cout << " Is Near Dead WG: " << isNearDeadWG() << std::endl;
}