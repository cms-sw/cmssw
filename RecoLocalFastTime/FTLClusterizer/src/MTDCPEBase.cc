#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"

#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEBase.h"

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"


#include <iostream>

using namespace std;

//-----------------------------------------------------------------------------
//  A constructor run for generic and templates
//
//-----------------------------------------------------------------------------
MTDCPEBase::MTDCPEBase(edm::ParameterSet const & conf,
		       const MTDGeometry& geom)
  : geom_(geom)
{
   fillDetParams();
}


//-----------------------------------------------------------------------------
//  Fill all variables which are constant for an event (geometry)
//-----------------------------------------------------------------------------
void MTDCPEBase::fillDetParams()
{
  auto const & dus = geom_.detUnits();
  unsigned m_detectors = dus.size();
  m_DetParams.resize(m_detectors);
  LogDebug("MTDCPEBase::fillDetParams():") <<"caching "<<m_detectors<<"MTD detectors"<<endl;
  for (unsigned i=0; i!=m_detectors;++i) 
    {
      auto & p=m_DetParams[i];
      p.theDet = dynamic_cast<const MTDGeomDetUnit*>(dus[i]);
      assert(p.theDet);
      
      p.theOrigin = p.theDet->surface().toLocal(GlobalPoint(0,0,0));
      
      //--- p.theDet->type() returns a GeomDetType, which implements subDetector()
      p.thePart = p.theDet->type().subDetector();
            
      //--- bounds() is implemented in BoundSurface itself.
      p.theThickness = p.theDet->surface().bounds().thickness();
      
      // Cache the det id for templates and generic erros
      p.theTopol = &(static_cast<const ProxyMTDTopology&>(p.theDet->topology()));
      assert(p.theTopol);
      p.theRecTopol = &(static_cast<const RectangularMTDTopology&>(p.theTopol->specificTopology()));    
      assert(p.theRecTopol);
      
      //--- The geometrical description of one module/plaquette
      std::pair<float,float> pitchxy = p.theRecTopol->pitch();
      p.thePitchX = pitchxy.first;	     // pitch along x
      p.thePitchY = pitchxy.second;	     // pitch along y
            
      LogDebug("MTDCPEBase::fillDetParams()") << "***** MTD LAYOUT *****"
					      << " thePart = " << p.thePart
					      << " theThickness = " << p.theThickness
					      << " thePitchX  = " << p.thePitchX
					      << " thePitchY  = " << p.thePitchY;
    }
}

//-----------------------------------------------------------------------------
//  One function to cache the variables common for one DetUnit.
//-----------------------------------------------------------------------------
void
MTDCPEBase::setTheClu( DetParam const & theDetParam, ClusterParam & theClusterParam ) const
{   
}

std::unique_ptr<MTDCPEBase::ClusterParam> MTDCPEBase::createClusterParam(const FTLCluster & cl) const
{
  return std::unique_ptr<ClusterParam>(new ClusterParam(cl));
}


//------------------------------------------------------------------------
MTDCPEBase::DetParam const & MTDCPEBase::detParam(const GeomDetUnit & det) const 
{
   auto i = det.index();
   assert(i<int(m_DetParams.size()));
   const DetParam & p = m_DetParams[i];
   return p;
}

LocalPoint
MTDCPEBase::localPosition(DetParam const & theDetParam, ClusterParam & theClusterParamBase) const
{
  //remember measurement point is row(col)+0.5f
  MeasurementPoint pos(theClusterParamBase.theCluster->x(),theClusterParamBase.theCluster->y());
  return theDetParam.theTopol->localPosition(pos);
}

LocalError
MTDCPEBase::localError(DetParam const & theDetParam,  ClusterParam & theClusterParamBase) const
{
  constexpr double one_over_twelve = 1./12.;
  MeasurementPoint pos(theClusterParamBase.theCluster->x(),theClusterParamBase.theCluster->y());
  MeasurementError simpleRect(one_over_twelve,0,one_over_twelve);
  return theDetParam.theTopol->localError(pos,simpleRect);
}

MTDCPEBase::TimeValue
MTDCPEBase::clusterTime(DetParam const & theDetParam, ClusterParam & theClusterParamBase) const
{
  return theClusterParamBase.theCluster->time();
}


MTDCPEBase::TimeValueError
MTDCPEBase::clusterTimeError(DetParam const & theDetParam, ClusterParam & theClusterParamBase) const
{
  return theClusterParamBase.theCluster->timeError();
}
