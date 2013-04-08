#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"


using namespace GeomDetEnumerators;


GEMEtaPartitionSpecs::GEMEtaPartitionSpecs( SubDetector rss, const std::string& name, const GEMSpecs& pars)
  :  GeomDetType(name,rss),_p(pars),_n(name)
{
  if (rss == GEM ){
    float b = _p[0];
    float B = _p[1];
    float h = _p[2];
    float r0 = h*(B+b)/(B-b);
    float striplength = h*2;
    float strips = _p[3];
    float pitch = (b+B)/strips;
    int nstrip =static_cast<int>(strips);
    _top = new TrapezoidalStripTopology(nstrip,pitch,striplength,r0);
  } else {
    _top = 0;
  }
}


GEMEtaPartitionSpecs::~GEMEtaPartitionSpecs()
{
  if (_top)
    delete _top;
  _top=0;
}


const Topology& 
GEMEtaPartitionSpecs::topology() const
{
  return *(_top);
}

const StripTopology& 
GEMEtaPartitionSpecs::specificTopology() const
{
  return *(_top);
}


const std::string&
GEMEtaPartitionSpecs::detName() const
{
  return _n;
}


