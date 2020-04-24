#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"


RPCRoll::RPCRoll(RPCDetId id, const BoundPlane::BoundPlanePointer& bp, RPCRollSpecs* rrs) :
  GeomDetUnit(bp), _id(id),_rrs(rrs)
{
  setDetId(id);
}

RPCRoll::~RPCRoll()
{
  delete _rrs; //Assume the roll owns it specs (specs are not shared)
}

const  RPCRollSpecs*
RPCRoll::specs() const
{
  return _rrs;
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

const RPCChamber* RPCRoll::chamber() const {
  return theCh;
}

int 
RPCRoll::nstrips() const
{
  return this->specificTopology().nstrips();
}

LocalPoint
RPCRoll::centreOfStrip(int strip) const
{
  float s = static_cast<float>(strip)-0.5;
  return this->specificTopology().localPosition(s);
}

LocalPoint
RPCRoll::centreOfStrip(float strip) const
{
  return this->specificTopology().localPosition(strip);
}

LocalError
RPCRoll::localError(float strip) const
{
  return this->specificTopology().localError(strip,1./sqrt(12.));
}

float
RPCRoll::strip(const LocalPoint& lp) const
{ 
  return this->specificTopology().strip(lp);

}

float
RPCRoll::localPitch(const LocalPoint& lp) const
{ 
  return this->specificTopology().localPitch(lp);

}

float
RPCRoll::pitch() const
{ 
  return this->specificTopology().pitch();

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



const StripTopology&
RPCRoll::specificTopology() const
{
  return _rrs->specificTopology();
}




void
RPCRoll::setChamber(const RPCChamber* ch)
{
  theCh = ch; 
}
