#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include <iostream>

LinkBoardSpec::LinkBoardSpec(bool m, int l, const ChamberLocationSpec & c)
    : theMaster(m), theLinkBoardNumInLink(l), theChamberSpec(c) { }

void LinkBoardSpec::print(int depth ) const 
{
  if (depth<0) return;
  std::string type = (theMaster) ? "master" : "slave";
  std::cout <<" LinkBoardSpec: " << std::endl
            <<" --->" <<type<<" linkBoardNumInLink: " << theLinkBoardNumInLink 
            << std::endl;
  theChamberSpec.print(--depth);
}
