#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"


GEMEtaPartition::GEMEtaPartition(GEMDetId id, BoundPlane::BoundPlanePointer bp, GEMEtaPartitionSpecs* rrs) :
  GeomDetUnit(bp), _id(id),_rrs(rrs)
{
  setDetId(id);
}

GEMEtaPartition::~GEMEtaPartition()
{
  delete _rrs; //Assume the roll owns it specs (specs are not shared)
}

const  GEMEtaPartitionSpecs*
GEMEtaPartition::specs() const
{
  return _rrs;
}

GEMDetId
GEMEtaPartition::id() const
{
  return _id;
}

const Topology&
GEMEtaPartition::topology() const
{
  return _rrs->topology();
}

const GeomDetType& 
GEMEtaPartition::type() const
{
  return (*_rrs);
}

/*
const GEMChamber* GEMEtaPartition::chamber() const {
  return theCh;
}
*/

int 
GEMEtaPartition::nstrips() const
{
  return this->specificTopology().nstrips();
}

LocalPoint
GEMEtaPartition::centreOfStrip(int strip) const
{
  float s = static_cast<float>(strip)-0.5;
  return this->specificTopology().localPosition(s);
}

LocalPoint
GEMEtaPartition::centreOfStrip(float strip) const
{
  return this->specificTopology().localPosition(strip);
}

LocalError
GEMEtaPartition::localError(float strip) const
{
  return this->specificTopology().localError(strip,1./sqrt(12.));
}

float
GEMEtaPartition::strip(const LocalPoint& lp) const
{ 
  return this->specificTopology().strip(lp);

}

float
GEMEtaPartition::localPitch(const LocalPoint& lp) const
{ 
  return this->specificTopology().localPitch(lp);

}

float
GEMEtaPartition::pitch() const
{ 
  return this->specificTopology().pitch();

}

const StripTopology&
GEMEtaPartition::specificTopology() const
{
  return _rrs->specificTopology();
}



/*
void
GEMEtaPartition::setChamber(const GEMChamber* ch)
{
  theCh = ch; 
}
*/
