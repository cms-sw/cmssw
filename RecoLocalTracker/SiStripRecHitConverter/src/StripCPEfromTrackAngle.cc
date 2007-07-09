#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::localParameters( const SiStripCluster & cl,const LocalTrajectoryParameters & ltp)const{
  //
  // get the det from the geometry
  //

  StripCPE::Param const & p = param(DetId(cl.geographicalId()));
  

 
  // ionisation length
   
  //float ionLen = std::min( trackDir.mag(), maxLength);

  //  const float par1=38.07;
  // const float par2=0.3184; 
  // const float par3=0.09828; 
  // const float P1 = par1 * p.thickness; 
  // const float P2 = par2; 
  // const float P3 = par3;


  //  drift*=(thickness/2);

  //calculate error form track angle


  LocalPoint  middlepoint = ltp.position();
  LocalVector trackDir = ltp.momentum()/ltp.momentum().mag();
  LocalPoint  position = p.topology->localPosition(cl.barycenter());



  if(trackDir.z()*p.drift.z() > 0.) trackDir *= -1.;


  if(trackDir.z() !=0.) {
    trackDir *= fabs(p.thickness/trackDir.z());
  } else {
    trackDir *= p.maxLength/trackDir.mag();
  }


      
  if(p.drift.z() == 0.) {
    //  if(drift.z() == 0.||cl.amplitudes().size()==1) {
    edm::LogError("StripCPE") <<"No drift towards anodes !!!";
    LocalError eresult = p.topology->localError(cl.barycenter(),1/12.);
    //  LocalPoint  result=LocalPoint(position.x()-drift.x()/2,position.y()-drift.y()/2,0);
    return std::make_pair(position-p.drift*(p.thickness/2),eresult);
  }	 




  // covered length along U
      
  LocalVector middleOfProjection = 0.5*(trackDir + p.drift);

  LocalPoint middlePointOnStrips = middlepoint + 0.5*p.drift;

  LocalPoint p1 = LocalPoint(middlePointOnStrips.x() + middleOfProjection.x()
			     ,middlePointOnStrips.y() + middleOfProjection.y());
  LocalPoint p2 = LocalPoint(middlePointOnStrips.x() - middleOfProjection.x()
			     ,middlePointOnStrips.y() - middleOfProjection.y());

  MeasurementPoint m1 = p.topology->measurementPosition(p1);
  MeasurementPoint m2 = p.topology->measurementPosition(p2);
  float u1 = m1.x();
  float u2 = m2.x();

  float uProj = std::min( float(fabs( u1 - u2)), float(p.nstrips));


//   float par1=38.07;
//   float par2=0.3184; 
//   float par3=0.09828; 
//   float P1 = par1 * thickness; 
//   float P2 = par2; 
//   float P3 = par3;
//   float uerr;

//   uerr =(uProj-P1)*(uProj-P1)*(P2-P3)/(P1*P1)+P3;

  const float P1=-0.314;
  const float P2=0.687;
  const float P3=0.294;
  
  float uerr=P1*uProj*exp(-uProj*P2)+P3;

  MeasurementError merror=MeasurementError( uerr*uerr, 0., 1./12.);
  LocalPoint result=LocalPoint(position.x()-0.5*p.drift.x(),position.y()-0.5*p.drift.y(),0);
  MeasurementPoint mpoint=p.topology->measurementPosition(result);
  LocalError eresult=p.topology->localError(mpoint,merror);
  return std::make_pair(result,eresult);
}


