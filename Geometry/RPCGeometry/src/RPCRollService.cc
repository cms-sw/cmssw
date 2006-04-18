#include "Geometry/RPCGeometry/interface/RPCRollService.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"



RPCRollService::RPCRollService() : roll_(0), top_(0)
{

}



RPCRollService::RPCRollService(RPCRoll* roll) : roll_(roll), top_(0)
{
}

RPCRollService::~RPCRollService()
{
}


int 
RPCRollService::nstrips()
{
  return this->topology()->nstrips();
}



LocalPoint 
RPCRollService::GlobalToLocalPoint(const GlobalPoint& gp)
{
  const BoundSurface& bSurface = roll_->surface();
  return bSurface.toLocal( gp );
}



GlobalPoint  
RPCRollService::LocalToGlobalPoint(const LocalPoint& lp)
{
  const BoundSurface& bSurface = roll_->surface();
  return bSurface.toGlobal( lp );
}

LocalPoint
RPCRollService::CentreOfStrip(int strip)
{
  float s = static_cast<float>(strip)-0.5;
  return this->topology()->localPosition(s);
}

LocalPoint
RPCRollService::CentreOfStrip(float strip)
{
  return this->topology()->localPosition(strip);
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



const StripTopology* 
RPCRollService::topology()
{
  if(!top_){
    if (this->isBarrel()){
      top_ = dynamic_cast<const RectangularStripTopology*>(&roll_->topology());
    }else{
      top_ = dynamic_cast<const TrapezoidalStripTopology*>(&roll_->topology());
    }
  }
  return top_;
}
