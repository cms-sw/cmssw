#include "Geometry/RPCGeometry/interface/RPCRollService.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"



RPCRollService::RPCRollService() : roll_(0)
{
}



RPCRollService::RPCRollService(RPCRoll* roll) : roll_(roll)
{
}

RPCRollService::~RPCRollService()
{
}


int 
RPCRollService::nstrips()
{
  int nstrs=0;
  if (this->isBarrel()){
    const RectangularStripTopology *top = 
      dynamic_cast<const RectangularStripTopology*>(&roll_->topology());
    nstrs=top->nstrips();
  }else{
    const TrapezoidalStripTopology *top = 
      dynamic_cast<const TrapezoidalStripTopology*>(&roll_->topology());
    nstrs=top->nstrips();
  }
  return nstrs;
}



GlobalPoint 
RPCRollService::GlobalToLocalPoint(const LocalPoint& lp)
{
  GlobalPoint gp;
  return gp;
}



LocalPoint  
RPCRollService::LocalToGlobalPoint(const GlobalPoint& gp)
{
  LocalPoint lp;
  return lp;
}


bool
RPCRollService::isBarrel()
{
  return ((roll_->id()).region()==0);
}  

bool 
RPCRollService::isForward()
{
  return (!this->isBarrel());
} 

