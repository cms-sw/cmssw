#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include <iostream>

void ChamberStripSpec::print( int depth ) const
{
  if (depth <0) return;
  std::cout << " ChamberStripSpec: " 
            << " pin: "<<cablePinNumber
            << ", chamber: "<<chamberStripNumber
            << ", CMS strip: "<<cmsStripNumber
            << std::endl;
}

