#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/TkRadialStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "boost/bind.hpp"
#include "boost/lambda/lambda.hpp"
#include <algorithm>
#include<cmath>
#include<cassert>

StripCPE::StripCPE( edm::ParameterSet & conf, 
		    const MagneticField& mag, 
		    const TrackerGeometry& geom, 
		    const SiStripLorentzAngle& LorentzAngle,
		    const SiStripConfObject& confObj,
		    const SiStripLatency& latency)
  : peakMode_(latency.singleReadOutMode() == 1),
    geom_(geom),
    magfield_(mag),
    LorentzAngleMap_(LorentzAngle)
{
  typedef std::map<std::string,SiStripDetId::ModuleGeometry> map_t;
  map_t modules;
  modules["IB1"]=SiStripDetId::IB1;  
  modules["IB2"]=SiStripDetId::IB2;
  modules["OB1"]=SiStripDetId::OB1;  
  modules["OB2"]=SiStripDetId::OB2;
  modules["W1A"]=SiStripDetId::W1A;    
  modules["W2A"]=SiStripDetId::W2A;  
  modules["W3A"]=SiStripDetId::W3A;  
  modules["W1B"]=SiStripDetId::W1B;
  modules["W2B"]=SiStripDetId::W2B;
  modules["W3B"]=SiStripDetId::W3B;
  modules["W4"] =SiStripDetId::W4;
  modules["W5"] =SiStripDetId::W5;
  modules["W6"] =SiStripDetId::W6;
  modules["W7"] =SiStripDetId::W7;
  
  const unsigned size = max_element( modules.begin(),modules.end(),
				     boost::bind(&map_t::value_type::second,boost::lambda::_1) < 
				     boost::bind(&map_t::value_type::second,boost::lambda::_2) )->second + 1;
  shift.resize(size);
  xtalk1.resize(size);
  xtalk2.resize(size);

  for(map_t::const_iterator it=modules.begin(); it!=modules.end(); it++) {
    const std::string 
      modeS(peakMode_?"Peak":"Deco"),
      shiftS( "shift_"  + it->first + modeS ),
      xtalk1S("xtalk1_" + it->first + modeS ),
      xtalk2S("xtalk2_" + it->first + modeS );

    if(!confObj.isParameter(shiftS)) throw cms::Exception("SiStripConfObject does not contain: ") << shiftS;
    if(!confObj.isParameter(xtalk1S)) throw cms::Exception("SiStripConfObject does not contain: ") << xtalk1S;
    if(!confObj.isParameter(xtalk2S)) throw cms::Exception("SiStripConfObject does not contain: ") << xtalk2S;

    shift[it->second] = confObj.get<double>(shiftS);
    xtalk1[it->second] = confObj.get<double>(xtalk1S);
    xtalk2[it->second] = confObj.get<double>(xtalk2S);
  }
  
  fillParams();

}

StripClusterParameterEstimator::LocalValues StripCPE::
localParameters( const SiStripCluster& cluster, const GeomDetUnit& det) const {
  StripCPE::Param const & p = param(det);
  const float barycenter = cluster.barycenter();
  const float fullProjection = p.coveredStrips( p.drift + LocalVector(0,0,-p.thickness), p.topology->localPosition(barycenter));
  const float strip = barycenter - 0.5f * (1.f-shift[p.moduleGeom]) * fullProjection;

  return std::make_pair( p.topology->localPosition(strip),
			 p.topology->localError(strip, 1.f/12.f) );
}

float StripCPE::Param::
coveredStrips(const LocalVector& lvec, const LocalPoint& lpos) const {  
  return topology->coveredStrips(lpos + 0.5f*lvec,lpos - 0.5f*lvec);
}

LocalVector StripCPE::
driftDirection(const StripGeomDetUnit* det) const { 
  LocalVector lbfield = (det->surface()).toLocal(magfield_.inTesla(det->surface().position()));  
  
  float tanLorentzAnglePerTesla = LorentzAngleMap_.getLorentzAngle(det->geographicalId().rawId());
  
  float dir_x = -tanLorentzAnglePerTesla * lbfield.y();
  float dir_y =  tanLorentzAnglePerTesla * lbfield.x();
  float dir_z =  1.f; // E field always in z direction
  
  return LocalVector(dir_x,dir_y,dir_z);
}


void 
StripCPE::fillParams() {  
  m_off = geom_.offsetDU(GeomDetEnumerators::TIB); // yes we know this
  auto const & dus = geom_.detUnits();
  m_Params.resize(dus.size()-m_off);
  for (auto i=m_off; i!=dus.size();++i) {
    auto & p= m_Params[i-m_off];
    const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(dus[i]);
    assert(stripdet->index()==int(i));
    assert(stripdet->geographicalId().subdetId()>1); // not pixel..

    const Bounds& bounds = stripdet->specificSurface().bounds();
    p.maxLength = std::sqrt( std::pow(bounds.length(),2.f)+std::pow(bounds.width(),2.f) );
    p.thickness = bounds.thickness();
    p.drift = driftDirection(stripdet) * p.thickness;
    p.topology=(StripTopology*)(&stripdet->topology());    
    p.nstrips = p.topology->nstrips(); 
    p.moduleGeom = SiStripDetId(stripdet->geographicalId()).moduleGeometry();
    
    const TkRadialStripTopology* rtop = dynamic_cast<const TkRadialStripTopology*>(&stripdet->specificType().specificTopology());
    p.pitch_rel_err2 = (rtop) 
      ? pow( 0.5f * rtop->angularWidth() * rtop->stripLength()/rtop->localPitch(LocalPoint(0,0,0)), 2.f) / 12.f
      : 0.f;
  }
}
