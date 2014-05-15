#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"


ME0EtaPartition::ME0EtaPartition(ME0DetId id, BoundPlane::BoundPlanePointer bp, ME0EtaPartitionSpecs* rrs) :
  GeomDetUnit(bp), id_(id),specs_(rrs)
{
  setDetId(id);
}

ME0EtaPartition::~ME0EtaPartition()
{
  delete specs_; //Assume the roll owns it specs (specs are not shared)
}

const Topology&
ME0EtaPartition::topology() const
{
  return specs_->topology();
}

const StripTopology&
ME0EtaPartition::specificTopology() const
{
  return specs_->specificTopology();
}

const Topology&
ME0EtaPartition::padTopology() const
{
  return specs_->padTopology();
}

const StripTopology&
ME0EtaPartition::specificPadTopology() const
{
  return specs_->specificPadTopology();
}

const GeomDetType& 
ME0EtaPartition::type() const
{
  return (*specs_);
}

int 
ME0EtaPartition::nstrips() const
{
  return this->specificTopology().nstrips();
}

LocalPoint
ME0EtaPartition::centreOfStrip(int strip) const
{
  float s = static_cast<float>(strip) - 0.5;
  return this->specificTopology().localPosition(s);
}

LocalPoint
ME0EtaPartition::centreOfStrip(float strip) const
{
  return this->specificTopology().localPosition(strip);
}

LocalError
ME0EtaPartition::localError(float strip) const
{
  return this->specificTopology().localError(strip, 1./sqrt(12.));
}

float
ME0EtaPartition::strip(const LocalPoint& lp) const
{ 
  return this->specificTopology().strip(lp);
}

float
ME0EtaPartition::localPitch(const LocalPoint& lp) const
{ 
  return this->specificTopology().localPitch(lp);
}

float
ME0EtaPartition::pitch() const
{ 
  return this->specificTopology().pitch();
}


int 
ME0EtaPartition::npads() const
{
  return specificPadTopology().nstrips();
}

LocalPoint
ME0EtaPartition::centreOfPad(int pad) const
{
  float p = static_cast<float>(pad) - 0.5;
  return specificPadTopology().localPosition(p);
}

LocalPoint
ME0EtaPartition::centreOfPad(float pad) const
{
  return specificPadTopology().localPosition(pad);
}

float
ME0EtaPartition::pad(const LocalPoint& lp) const
{ 
  return specificPadTopology().strip(lp);
}

float
ME0EtaPartition::localPadPitch(const LocalPoint& lp) const
{ 
  return specificPadTopology().localPitch(lp);
}

float
ME0EtaPartition::padPitch() const
{ 
  return specificPadTopology().pitch();
}


float
ME0EtaPartition::padOfStrip(int strip) const
{
  LocalPoint c_o_s = centreOfStrip(strip);
  return pad(c_o_s);
}

int
ME0EtaPartition::firstStripInPad(int pad) const
{
  float p = static_cast<float>(pad) - 0.9999;
  LocalPoint lp = specificPadTopology().localPosition(p);
  return static_cast<int>(strip(lp)) + 1;
}

int
ME0EtaPartition::lastStripInPad(int pad) const
{
  float p = static_cast<float>(pad) - 0.0001;
  LocalPoint lp = specificPadTopology().localPosition(p);
  return static_cast<int>(strip(lp)) + 1;
}

