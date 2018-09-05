#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"


GEMEtaPartition::GEMEtaPartition(GEMDetId id, const BoundPlane::BoundPlanePointer& bp, GEMEtaPartitionSpecs* rrs) :
  GeomDet(bp), id_(id),specs_(rrs)
{
  setDetId(id);
}

GEMEtaPartition::~GEMEtaPartition()
{
  delete specs_; //Assume the roll owns it specs (specs are not shared)
}

const Topology&
GEMEtaPartition::topology() const
{
  return specs_->topology();
}

const StripTopology&
GEMEtaPartition::specificTopology() const
{
  return specs_->specificTopology();
}

const Topology&
GEMEtaPartition::padTopology() const
{
  return specs_->padTopology();
}

const StripTopology&
GEMEtaPartition::specificPadTopology() const
{
  return specs_->specificPadTopology();
}

const GeomDetType& 
GEMEtaPartition::type() const
{
  return (*specs_);
}

int 
GEMEtaPartition::nstrips() const
{
  return this->specificTopology().nstrips();
}

LocalPoint
GEMEtaPartition::centreOfStrip(int strip) const
{
  float s = static_cast<float>(strip) + 0.5f;
  return this->specificTopology().localPosition(s);
}

LocalPoint
GEMEtaPartition::centreOfStrip(float strip) const
{
  return this->specificTopology().localPosition(strip);
}

LocalError
GEMEtaPartition::localError(float strip, float cluster_size) const
{
  return this->specificTopology().localError(strip, cluster_size*cluster_size/12.);
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


int 
GEMEtaPartition::npads() const
{
  return specificPadTopology().nstrips();
}

LocalPoint
GEMEtaPartition::centreOfPad(int pad) const
{
  float p = static_cast<float>(pad) + 0.5f;
  return specificPadTopology().localPosition(p);
}

LocalPoint
GEMEtaPartition::centreOfPad(float pad) const
{
  return specificPadTopology().localPosition(pad);
}

float
GEMEtaPartition::pad(const LocalPoint& lp) const
{ 
  return specificPadTopology().strip(lp);
}

float
GEMEtaPartition::localPadPitch(const LocalPoint& lp) const
{ 
  return specificPadTopology().localPitch(lp);
}

float
GEMEtaPartition::padPitch() const
{ 
  return specificPadTopology().pitch();
}


float
GEMEtaPartition::padOfStrip(int strip) const
{
  LocalPoint c_o_s = centreOfStrip(strip);
  return pad(c_o_s);
}

int
GEMEtaPartition::firstStripInPad(int pad) const
{
  float p = static_cast<float>(pad) - 0.9999f;
  LocalPoint lp = specificPadTopology().localPosition(p);
  return static_cast<int>(strip(lp));
}

int
GEMEtaPartition::lastStripInPad(int pad) const
{
  float p = static_cast<float>(pad) - 0.0001f;
  LocalPoint lp = specificPadTopology().localPosition(p);
  return static_cast<int>(strip(lp));
}

