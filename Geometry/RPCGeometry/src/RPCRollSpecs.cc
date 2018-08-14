#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"


using namespace GeomDetEnumerators;


RPCRollSpecs::RPCRollSpecs( SubDetector rss, const std::string& name, const RPCSpecs& pars)
  :  GeomDetType(name,rss),_p(pars),_n(name)
{
  if (rss == RPCBarrel ){
    int nstrip =static_cast<int>( _p[2]);
    float pitch = _p[0]*2/_p[2];
    float striplength = _p[1]*2;
    _top = new RectangularStripTopology(nstrip,pitch,striplength);
  }else if (rss == RPCEndcap ){
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
    _top = nullptr;
  }
}


RPCRollSpecs::~RPCRollSpecs()
{
  if (_top)
    delete _top;
  _top=nullptr;
}


const Topology& 
RPCRollSpecs::topology() const
{
  return *(_top);
}

const StripTopology& 
RPCRollSpecs::specificTopology() const
{
  return *(_top);
}


const std::string&
RPCRollSpecs::detName() const
{
  return _n;
}
