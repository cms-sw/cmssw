#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
//typedef std::pair<LocalPoint,LocalError>  LocalValues;


StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::localParameters( const SiStripCluster & cl,const LocalTrajectoryParameters & ltp)const{
  //
  // get the det from the geometry
  //

  LocalPoint middlepoint = ltp.position();
  LocalVector atrackUnit = ltp.momentum()/ltp.momentum().mag();

  DetId detId(cl.geographicalId());
  const GeomDetUnit *  det = geom_->idToDetUnit(detId);

  LocalPoint position;
  LocalError eresult;
  LocalVector drift=LocalVector(0,0,1);
  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
  //  DetId detId(det.geographicalId());
  const StripTopology &topol=(StripTopology&)stripdet->topology();
  position = topol.localPosition(cl.barycenter());
  
  
  drift= driftDirection(stripdet);
  float thickness=stripdet->surface().bounds().thickness();

  //  drift*=(thickness/2);

  //calculate error form track angle


  LocalVector trackDir = atrackUnit;
      
  if(drift.z() == 0.) {
    //  if(drift.z() == 0.||cl.amplitudes().size()==1) {
    edm::LogError("StripCPE") <<"No drift towards anodes !!!";
    eresult = topol.localError(cl.barycenter(),1/12.);
    //  LocalPoint  result=LocalPoint(position.x()-drift.x()/2,position.y()-drift.y()/2,0);
    return std::make_pair(position-drift*(thickness/2),eresult);
  }	 

  if(trackDir.z()*drift.z() > 0.) trackDir *= -1.;

  const Bounds& bounds = stripdet->surface().bounds();


  //  float maxLength = sqrt( pow(double(bounds.length()),2.)+pow(double(bounds.width()),2.) );
  float maxLength = sqrt( bounds.length()*bounds.length()+bounds.width()*bounds.width());
  drift *= fabs(thickness/drift.z());       
  if(trackDir.z() !=0.) {
    trackDir *= fabs(thickness/trackDir.z());
  } else {
    trackDir *= maxLength/trackDir.mag();
  }

  // covered length along U
      
  LocalVector middleOfProjection = 0.5*(trackDir + drift);

  LocalPoint middlePointOnStrips = middlepoint + 0.5*drift;

  LocalPoint p1 = LocalPoint(middlePointOnStrips.x() + middleOfProjection.x()
			     ,middlePointOnStrips.y() + middleOfProjection.y());
  LocalPoint p2 = LocalPoint(middlePointOnStrips.x() - middleOfProjection.x()
			     ,middlePointOnStrips.y() - middleOfProjection.y());

  MeasurementPoint m1 = topol.measurementPosition(p1);
  MeasurementPoint m2 = topol.measurementPosition(p2);
  float u1 = m1.x();
  float u2 = m2.x();
  int nstrips = topol.nstrips(); 
  float uProj = std::min( float(fabs( u1 - u2)), float(nstrips));

  // ionisation length
   
  //float ionLen = std::min( trackDir.mag(), maxLength);

//   float par1=38.07;
//   float par2=0.3184; 
//   float par3=0.09828; 
//   float P1 = par1 * thickness; 
//   float P2 = par2; 
//   float P3 = par3;
//   float uerr;

//   uerr =(uProj-P1)*(uProj-P1)*(P2-P3)/(P1*P1)+P3;
  float P1=-0.314;
  float P2=0.687;
  float P3=0.294;
  float uerr;
  uerr=P1*uProj*exp(-uProj*P2)+P3;

  
  MeasurementError merror=MeasurementError( uerr*uerr, 0., 1./12.);
  LocalPoint result=LocalPoint(position.x()-drift.x()/2,position.y()-drift.y()/2,0);
  MeasurementPoint mpoint=topol.measurementPosition(result);
  eresult=topol.localError(mpoint,merror);
return std::make_pair(result,eresult);
}


