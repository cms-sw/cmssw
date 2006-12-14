#include "L1Trigger/RBCEmulator/interface/RBCPolicy.h"
#include "L1Trigger/RBCEmulator/src/RBCLogic.h"
#include "L1Trigger/RBCEmulator/src/RBCPatternLogic.h"
#include "L1Trigger/RBCEmulator/src/RBCChamberORLogic.h"


RBCPolicy::RBCPolicy(RBCPolicy::Policy p) : pol(p), l(0)    
{
}


RBCPolicy::~RBCPolicy()
{ 
  std::cout<<"bye policy"<<std::endl;
  if (l)
    delete l;
  l=0;
}



std::string 
RBCPolicy::message()
{
    if (pol == Patterns)
      return "Patterns";
    else if ( pol == ChamberOR )
      return "ChamberOR";
    else
      return "No idea";
}
  


RBCLogic* 

RBCPolicy::instance()
{
  if (l)
    delete l;
  l=0;
  if (pol == Patterns)
    l = new RBCPatternLogic();
  else if ( pol == ChamberOR )
    l =  new RBCChamberORLogic();
  return l;
}
