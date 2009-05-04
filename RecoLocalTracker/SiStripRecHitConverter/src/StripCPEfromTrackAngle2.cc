#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle2.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <memory>
#include <string>

using namespace std;

StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle2::localParameters( const SiStripCluster & cl,const LocalTrajectoryParameters & ltp)const{

  StripCPE::Param const & p = param(DetId(cl.geographicalId()));
  
  LocalPoint middlepoint = ltp.position();
  LocalVector trackDir = ltp.momentum()/ltp.momentum().mag();
  LocalPoint  position = p.topology->localPosition(cl.barycenter());
  
  if(p.drift.z() == 0.) {
    edm::LogError("StripCPE") <<"No drift towards anodes !!!";
    LocalError eresult = p.topology->localError(cl.barycenter(),1/12.);
    return std::make_pair(position-p.drift*(p.thickness/2),eresult);
  }	 

  if(trackDir.z()*p.drift.z() > 0.) trackDir *= -1.;

  if(trackDir.z() !=0.) {
    trackDir *= fabs(p.thickness/trackDir.z());
  } else {
    trackDir *= p.maxLength/trackDir.mag();
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

  int clusterWidth = cl.amplitudes().size();

  int expectedWidth = 1;
  if (u1<u2) expectedWidth = 1 + int(u2) -int(u1);
  if (u1>u2) expectedWidth = 1 + int(u1) -int(u2);
  if (u1==u2) expectedWidth = 1;
  //  expectedWidth = 1 + int(u2) -int(u1);


  // Coefficients P0, P1 & P2 of quadratic parametrization of 
  // resolution in terms of track width.
  // The "nbin" bins each correspond to different classes of clusters that
  // need different parametrizations (iopt=0,1,2 below).

  float projectionOnU ;
  projectionOnU = uProj;
  const unsigned int nbins = 3;
  //const float P0[nbins] =  { 0.329,   0.069,   0.549};
  //const float P1[nbins] =  {-0.088,   0.049,  -0.619};
  //const float P2[nbins] =  {-0.115,   0.004,   0.282};

  // Good param
  /*
  const float P0[nbins] =  {2.45226e-01,9.09088e-02, 2.43403e-01 };
  const float P1[nbins] =  {-3.50310e-02,-1.37922e-02,-1.93286e-01  };
  const float P2[nbins] =  {-1.14758e-01,1.31774e-02,7.68252e-02};
  */
  const float P0[nbins] =  {2.62868e-01 ,2.07353e-01,2.56743e-01};//best
  const float P1[nbins] =  {-1.11246e-01 ,-1.44337e-01,-1.84289e-01};//best
  const float P2[nbins] =  {-3.86084e-02,4.76344e-02,6.69619e-02};//best
  //

    //
  const float minError = 0.1;

  // Coefficients of linear parametrization in terms of cluster
  // width that is used for bad clusters (iopt=999 below).
  //const float q0 = 1.80;
  //const float q1 = 2.83;

  // Good param 
  const float q0 = 9.07253e-01; //best
  const float q1 = 3.10393e+00;//best
  //const float q0 = 3.83120e-01;
  //const float q1 = 1.80999e+00;

  unsigned int iopt;
  if (clusterWidth > expectedWidth + 2) {
    // This cluster is not good - much wider than expected ...
    // (Happens for about 3% of clusters in typical events).
    iopt = 999;
  } else if (expectedWidth == 1) {
    // In this case, charge sharing doesn't improve resolution.
    // (Happens for about 70% of clusters in typical events).
    iopt = 0;
  } else if (clusterWidth <= expectedWidth) {
    // Resolution rather good in this case.
    // (Happens for about 18% of clusters in typical events).
    iopt = 1;
  } else {
    // Can happen due to inter-strip coupling or noise.
    // (Happens for about 6% of clusters in typical events).
    iopt = 2;
  }

  float uerr;
  if (iopt == 999) {
    // Cluster much wider than expected (delta ray ?), so 
    // assign large error.
    uerr = q0*(clusterWidth - q1)/sqrt(12.);

  } else if (iopt < nbins) {
   // Quadratic parametrization.
    uerr = P0[iopt] + P1[iopt]*projectionOnU 
                    + P2[iopt]*projectionOnU*projectionOnU;


  } else {
    cout<<"Bug in StripCPEfromTrackAngle2"<<endl;
    exit(9);
  }



  // For the sake of stability, avoid very small resolution.
  if (uerr < minError) uerr = minError;
  
  short firstStrip = cl.firstStrip();

  short lastStrip = firstStrip + clusterWidth - 1;
  
  // If cluster reaches edge of sensor, inflate its error.
    if (firstStrip == 0 || lastStrip == p.nstrips - 1) {
      uerr += 0.5*(1 + projectionOnU);
  }
     
    //if((cl.amplitudes().size() - (uProj+2.5)) > 1) uerr=cl.amplitudes().size()  * (1./sqrt(12.));
  
  MeasurementError merror=MeasurementError( uerr*uerr, 0., 1./12.);
  LocalPoint result=LocalPoint(position.x()-0.5*p.drift.x(),position.y()-0.5*p.drift.y(),0);
  MeasurementPoint mpoint=p.topology->measurementPosition(result);
  LocalError eresult=p.topology->localError(mpoint,merror);


  return std::make_pair(result,eresult);
  
}


