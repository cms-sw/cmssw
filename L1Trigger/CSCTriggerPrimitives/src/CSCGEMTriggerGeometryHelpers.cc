#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMTriggerGeometryHelpers.h"

using namespace cscgemtriggeom;

/*
    // loop on all wiregroups to create a LUT <WG,rollMin,rollMax>
    const int numberOfWG(cscChamber->layer(1)->geometry()->numberOfWireGroups());
    std::cout <<"detId " << cscChamber->id() << std::endl;
    for (int i = 0; i< numberOfWG; ++i){
      auto lpc(cscChamber->layer(1)->geometry()->localCenterOfWireGroup(i));
      auto gpc(cscChamber->layer(1)->toGlobal(lpc));
      auto eta(gpc.eta());
      std::cout << "{";
      if (i<10) std::cout << " ";
      std::cout << i << "," << std::fixed << std::setprecision(3) << std::abs(eta) << "},";
      if ((i+1)%8==0 and i!=1) std::cout << std::endl;
    }
*/
