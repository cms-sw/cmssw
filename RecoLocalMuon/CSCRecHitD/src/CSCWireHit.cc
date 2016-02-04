#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>
#include <iostream>

CSCWireHit::CSCWireHit() :
  theDetId(),
  theWireHitPosition(),
  theWgroups(),
  theWireHitTmax(),
  theDeadWG(),
  theTimeBinsOn(0)
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
                        const short int& deadWG,
                        const std::vector <int>& timeBinsOn ) :
  theDetId( id ), 
  theWireHitPosition( wHitPos ),
  theWgroups( wgroups ),
  theWireHitTmax ( tmax ),
  theDeadWG ( deadWG ),
  theTimeBinsOn( timeBinsOn )
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
   std::cout << " Dead WG: " << deadWG() << std::endl;
   std::cout << " Time bins on: ";
   for (int i=0; i<(int) timeBinsOn().size(); i++) std::cout << timeBinsOn()[i] << " ";
   std::cout << std::endl;
}
