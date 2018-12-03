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
      //      assert(p.theDet->index()==int(i));
      
      p.theOrigin = p.theDet->surface().toLocal(GlobalPoint(0,0,0));
      
      //--- p.theDet->type() returns a GeomDetType, which implements subDetector()
      p.thePart = p.theDet->type().subDetector();
      
      //--- The location in of this DetUnit in a cyllindrical coord system (R,Z)
      //--- The call goes via BoundSurface, returned by p.theDet->surface(), but
      //--- position() is implemented in GloballyPositioned<> template
      //--- ( BoundSurface : Surface : GloballyPositioned<float> )
      //p.theDetR = p.theDet->surface().position().perp();  //Not used, AH
      //p.theDetZ = p.theDet->surface().position().z();  //Not used, AH
      //--- Define parameters for chargewidth calculation
      
      //--- bounds() is implemented in BoundSurface itself.
      p.theThickness = p.theDet->surface().bounds().thickness();
      
      // Cache the det id for templates and generic erros
      p.theTopol = &(static_cast<const ProxyMTDTopology&>(p.theDet->topology()));
      assert(p.theTopol);
      p.theRecTopol = &(static_cast<const RectangularMTDTopology&>(p.theTopol->specificTopology()));    
      assert(p.theRecTopol);
      
      //--- The geometrical description of one module/plaquette
      //p.theNumOfRow = p.theRecTopol->nrows();	// rows in x //Not used, AH. PM: leave commented out.
      //p.theNumOfCol = p.theRecTopol->ncolumns();	// cols in y //Not used, AH. PM: leave commented out.
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

MTDCPEBase::ClusterParam* MTDCPEBase::createClusterParam(const FTLCluster & cl) const
{
   return new ClusterParam(cl);
}


//------------------------------------------------------------------------
MTDCPEBase::DetParam const & MTDCPEBase::detParam(const GeomDetUnit & det) const 
{
   auto i = det.index();
   //cout << "get parameters of detector " << i << endl;
   assert(i<int(m_DetParams.size()));
   //if (i>=int(m_DetParams.size())) m_DetParams.resize(i+1);  // should never happen!
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
  double one_over_twelve = 1./12.;
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
