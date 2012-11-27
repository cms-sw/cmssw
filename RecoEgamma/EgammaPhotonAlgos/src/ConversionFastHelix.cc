#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionFastHelix.h"
#include "RecoTracker/TkSeedGenerator/interface/FastLine.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"
#include "FWCore/Utilities/interface/isFinite.h"


#include <cfloat>

ConversionFastHelix::ConversionFastHelix(const GlobalPoint& outerHit,
		     const GlobalPoint& middleHit,
					 const GlobalPoint& aVertex,
					 const MagneticField* field ) : 
  theOuterHit(outerHit),
  theMiddleHit(middleHit),
  theVertex(aVertex),
  theCircle(outerHit,
	    middleHit,
	    aVertex),
  mField(field) {
  
  validStateAtVertex=false;


  makeHelix();
 
  
}


void ConversionFastHelix::makeHelix()  {


  if(   theCircle.isValid()) {
     theHelix_=helixStateAtVertex();
  } else {
     theHelix_= straightLineStateAtVertex();
  }
  

}


FreeTrajectoryState ConversionFastHelix::stateAtVertex()  {

  return theHelix_;

}


FreeTrajectoryState ConversionFastHelix::helixStateAtVertex()  {
  
  
  
  GlobalPoint pMid(theMiddleHit);
  GlobalPoint v(theVertex);
  FTS atVertex;
  
  double dydx = 0.;
  double pt = 0., px = 0., py = 0.;
  
  //remember (radius rho in cm):
  //rho = 
  //100. * pt * 
  //(10./(3.*MagneticField::inTesla(GlobalPoint(0., 0., 0.)).z()));
  
  double rho = theCircle.rho();
  pt = 0.01 * rho * (0.3*mField->inTesla(GlobalPoint(0,0,0)).z());
  
  // (py/px)|x=v.x() = (dy/dx)|x=v.x()
  //remember:
  //y(x) = +-sqrt(rho^2 - (x-x0)^2) + y0 
  //y(x) =  sqrt(rho^2 - (x-x0)^2) + y0  if y(x) >= y0 
  //y(x) = -sqrt(rho^2 - (x-x0)^2) + y0  if y(x) < y0
  //=> (dy/dx) = -(x-x0)/sqrt(Q)  if y(x) >= y0
  //   (dy/dx) =  (x-x0)/sqrt(Q)  if y(x) < y0
  //with Q = rho^2 - (x-x0)^2
  
  
  double arg=rho*rho - ( (v.x()-theCircle.x0())*(v.x()-theCircle.x0()) );
  
  if ( arg >= 0 ) { 
    
    
    //  double root = sqrt(  rho*rho - ( (v.x()-theCircle.x0())*(v.x()-theCircle.x0()) )  );
    double root = sqrt(  arg );
    
    if((v.y() - theCircle.y0()) > 0.)
      dydx = -(v.x() - theCircle.x0()) / root;
    else
      dydx = (v.x() - theCircle.x0()) / root;
    
    px = pt/sqrt(1. + dydx*dydx);
    py = px*dydx;
    // check sign with scalar product
    if(px*(pMid.x() - v.x()) + py*(pMid.y() - v.y()) < 0.) {
      px *= -1.;
      py *= -1.;
    } 
    
    //std::cout << " ConversionFastHelix:helixStateAtVertex  rho " << rho  << " pt " << pt  << " v " <<  v << " theCircle.x0() " <<theCircle.x0() << " theCircle.y0() "  <<  theCircle.y0() << " v.x()-theCircle.x0() "  << v.x()-theCircle.x0() << " rho^2 " << rho*rho << "  v.x()-theCircle.x0()^2 " <<   (v.x()-theCircle.x0())*(v.x()-theCircle.x0()) <<  " root " << root << " arg " << arg <<  " dydx " << dydx << std::endl;
    //calculate z0, pz
    //(z, R*phi) linear relation in a helix
    //with R, phi defined as radius and angle w.r.t. centre of circle
    //in transverse plane
    //pz = pT*(dz/d(R*phi)))
    
    FastLine flfit(theOuterHit, theMiddleHit, theCircle.rho());
   
    
    
    double z_0 = 0; 
    
    //std::cout << " ConversionFastHelix:helixStateAtVertex  flfit.n2() " <<  flfit.n2() << " flfit.c() " << flfit.c() << " flfit.n2() " << flfit.n2() << std::endl;
    if ( flfit.n2() !=0 && !edm::isNotFinite( flfit.c()) && !edm::isNotFinite(flfit.n2())   ) {
      //  std::cout << " Accepted " << std::endl;
      z_0 = -flfit.c()/flfit.n2();
      double dzdrphi = -flfit.n1()/flfit.n2();
      double pz = pt*dzdrphi;
      
      //get sign of particle
      
      GlobalVector magvtx=mField->inTesla(v);
      TrackCharge q = 
	((theCircle.x0()*py - theCircle.y0()*px) / 
	 (magvtx.z()) < 0.) ? 
	-1 : 1;
      

      AlgebraicSymMatrix55 C = AlgebraicMatrixID();
      //MP
      
      atVertex = FTS(GlobalTrajectoryParameters(GlobalPoint(v.x(), v.y(), z_0),
						GlobalVector(px, py, pz),
						q,
						mField),
		     CurvilinearTrajectoryError(C));
      
      //std::cout << " ConversionFastHelix:helixStateAtVertex globalPoint " << GlobalPoint(v.x(), v.y(), z_0) << " GlobalVector " << GlobalVector(px, py, pz)  << " q " << q << " MField " << mField->inTesla(v) << std::endl;
      //std::cout << " ConversionFastHelix:helixStateAtVertex atVertex.transverseCurvature() " << atVertex.transverseCurvature() << std::endl;
      if( atVertex.transverseCurvature() !=0 ) {
	
	validStateAtVertex=true;    
	
	//std::cout << " ConversionFastHelix:helixStateAtVertex validHelixStateAtVertex status " << validStateAtVertex << std::endl;
	return atVertex;
      }else
	return atVertex;
    } else {
      //std::cout << " ConversionFastHelix:helixStateAtVertex not accepted  validHelixStateAtVertex status  " << validStateAtVertex << std::endl;
      return atVertex;
    }
    
    
    
  } else {
    
    //std::cout << " ConversionFastHelix:helixStateAtVertex not accepted because arg <0 validHelixStateAtVertex status  " << validStateAtVertex << std::endl;
    return atVertex;
  }
  


  
  
}

FreeTrajectoryState ConversionFastHelix::straightLineStateAtVertex() {

  FTS atVertex;

  //calculate FTS assuming straight line...

  GlobalPoint pMid(theMiddleHit);
  GlobalPoint v(theVertex);

  double dydx = 0.;
  double pt = 0., px = 0., py = 0.;
  
  if(fabs(theCircle.n1()) > 0. || fabs(theCircle.n2()) > 0.)
    pt = 1.e+4;// 10 TeV //else no pt
  if(fabs(theCircle.n2()) > 0.) {
    dydx = -theCircle.n1()/theCircle.n2(); //else px = 0 
  }

  if ( pt==0 && dydx==0. ) {
    validStateAtVertex=false;
    return atVertex; 
  }
  px = pt/sqrt(1. + dydx*dydx);
  py = px*dydx;

  // check sign with scalar product
  if (px*(pMid.x() - v.x()) + py*(pMid.y() - v.y()) < 0.) {
    px *= -1.;
    py *= -1.;
  } 

  //calculate z_0 and pz at vertex using weighted mean
  //z = z(r) = z0 + (dz/dr)*r
  //tan(theta) = dr/dz = (dz/dr)^-1
  //theta = atan(1./dzdr)
  //p = pt/sin(theta)
  //pz = p*cos(theta) = pt/tan(theta) 


  FastLine flfit(theOuterHit, theMiddleHit);

  double z_0 = 0;
  if (flfit.n2() !=0  && !edm::isNotFinite( flfit.c()) && !edm::isNotFinite(flfit.n2())   ) {
    z_0 = -flfit.c()/flfit.n2();

    double dzdr = -flfit.n1()/flfit.n2();
    double pz = pt*dzdr; 
    
    TrackCharge q = 1;
 
    atVertex = FTS(GlobalPoint(v.x(), v.y(), z_0),
		   GlobalVector(px, py, pz),
		   q,
		   mField
		   );

    //    std::cout << "  ConversionFastHelix::straightLineStateAtVertex curvature " << atVertex.transverseCurvature() << "   signedInverseMomentum " << atVertex.signedInverseMomentum() << std::endl;
    if ( atVertex.transverseCurvature() == -0 ) {
      return atVertex;
    } else {
      validStateAtVertex=true;        
      return atVertex;
    }
  
  } else {


    return atVertex;
  
  }

}
