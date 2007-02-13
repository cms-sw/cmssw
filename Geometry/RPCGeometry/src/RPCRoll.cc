#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"


RPCRoll::RPCRoll( BoundPlane* bp, RPCRollSpecs* rrs, RPCDetId id) :
  GeomDetUnit(bp), top_(0), _id(id),_rrs(rrs)
{
  
}

RPCRoll::~RPCRoll()
{

}

const  RPCRollSpecs*
RPCRoll::specs() const
{
  return _rrs;
}

DetId
RPCRoll::geographicalId() const
{
  return _id;
}

RPCDetId
RPCRoll::id() const
{
  return _id;
}

const Topology&
RPCRoll::topology() const
{
  return _rrs->topology();
}

const GeomDetType& 
RPCRoll::type() const
{
  return (*_rrs);
}


int 
RPCRoll::nstrips() const
{
  return this->striptopology()->nstrips();
}

LocalPoint
RPCRoll::centreOfStrip(int strip) const
{
  float s = static_cast<float>(strip)-0.5;
  return this->striptopology()->localPosition(s);
}

LocalPoint
RPCRoll::centreOfStrip(float strip) const
{
  return this->striptopology()->localPosition(strip);
}

LocalError
RPCRoll::localError(float strip) const
{
  return this->striptopology()->localError(strip,1./sqrt(12.));
}

float
RPCRoll::strip(const LocalPoint& lp) const
{ 
  return this->striptopology()->strip(lp);

}

float
RPCRoll::localPitch(const LocalPoint& lp) const
{ 
  return this->striptopology()->localPitch(lp);

}

float
RPCRoll::pitch() const
{ 
  return this->striptopology()->pitch();

}

bool
RPCRoll::isBarrel() const
{
  return ((this->id()).region()==0);
}  

bool 
RPCRoll::isForward() const

{
  return (!this->isBarrel());
} 



const StripTopology* 
RPCRoll::striptopology() const
{
  if(!top_){
    if (this->isBarrel()){
      top_ = dynamic_cast<const RectangularStripTopology*>(&this->topology());
    }else{
      top_ = dynamic_cast<const TrapezoidalStripTopology*>(&this->topology());
      
    }
  }
  return top_;
}
