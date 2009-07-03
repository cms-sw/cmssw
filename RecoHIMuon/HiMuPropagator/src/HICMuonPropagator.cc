#include "RecoHIMuon/HiMuPropagator/interface/HICMuonPropagator.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
 
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Vector/ThreeVector.h"
#include <CLHEP/Vector/LorentzVector.h>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
//#define MUPROPAGATOR_DEBUG
namespace cms {
TrajectoryStateOnSurface
             HICMuonPropagator::propagate(const FreeTrajectoryState& fts,
  			                  const Cylinder& surface) const
  {
#ifdef MUPROPAGATOR_DEBUG
    cout<<"MuPropagator::Start propagation"<<endl;
#endif
// From FreeTrajectoryState to the vertex.

  GlobalPoint x = fts.parameters().position();
  GlobalVector p = fts.parameters().momentum();
  double px = p.x(); double py = p.y(); double pz = p.z();
  double aCharge = fts.charge();
  AlgebraicSymMatrix55 e = fts.curvilinearError().matrix();
  double dfcalc,phnext,zdet;
//  double pt = p.perp();
  double a = p.perp()/pz;

  double b = -a*theHICConst->zvert; 
 
  double phiold=x.phi();
  if(x.phi()<0.) phiold=twopi+x.phi();
    
#ifdef MUPROPAGATOR_DEBUG  
  cout<< "MuPropagator::xold=" << x<<" p="<<p<<endl;
  cout<<"MuPropagator::Propagate to cylinder="<<surface.radius()<<" pt="<<pt<<
  " line parameters="<<a<<" "<<b<<endl;
#endif 
 
// Propagate on surface:phidet

  dfcalc = aCharge*0.006*abs(x.perp()-surface.radius())/p.perp();
  phnext = phiold+dfcalc;

  if(phnext>twopi) phnext = phnext-twopi;
  if(phnext<0.) phnext = phnext+twopi;
  
// Propagate Zdet

  zdet = (surface.radius()-b)/a;
  
// New state
  GlobalPoint xnew(surface.radius()*cos(phnext),surface.radius()*sin(phnext),zdet);
  GlobalVector pnew(px*cos(dfcalc)-py*sin(dfcalc),px*sin(dfcalc)+py*cos(dfcalc),pz);
#ifdef MUPROPAGATOR_DEBUG
  cout<< "almost empty propagator for the moment phnext,zdet="<<phnext<<" "<<zdet<<endl;
  cout<<"New coordinates="<<xnew<<endl;
  cout<<"New momentum="<<pnew<<endl;
#endif  

  TrajectoryStateOnSurface tsos(
                                 GlobalTrajectoryParameters(xnew, pnew, (int)aCharge, field),
                                 CurvilinearTrajectoryError(e), surface
				 );
     return tsos;
  }

TrajectoryStateOnSurface
             HICMuonPropagator::propagate(const FreeTrajectoryState& fts,
  			                  const Plane& surface) const
  {


//
//  Check if it is detector or layer 
//  if(surface.position().perp()>0.000001) {
//  HICTrajectoryCorrector* theNewPropagator = new HICTrajectoryCorrector();
//  TrajectoryStateOnSurface tsos = theCorrector->propagate(fts,surface);
//  delete theNewPropagator;
//  return tsos;
//  } else {
//  Check if it is forward pixel detector
// if(abs(surface.position().z())>0. && abs(surface.position().z())<50.) {
//  GtfPropagator* theNewPropagator = new GtfPropagator(oppositeToMomentum);
//  TrajectoryStateOnSurface tsos = theCorrector->propagate(fts,surface);
//  delete theNewPropagator;
//  return tsos;
// }
// }
//
  double dfcalc,phnext,rdet;
  
#ifdef MUPROPAGATOR_DEBUG
    cout<<"MuPropagator::Start propagation"<<endl;
#endif
// Information from previous layer
//
  GlobalPoint x = fts.parameters().position();
  GlobalVector p = fts.parameters().momentum();
  double px = p.x(); double py = p.y(); double pz = p.z();
  double aCharge = fts.charge();
  AlgebraicSymMatrix55 e = fts.curvilinearError().matrix();
  double phiold=x.phi();
  if(x.phi()<0.) phiold=twopi+x.phi();

#ifdef MUPROPAGATOR_DEBUG  
  cout<< "MuPropagator::xold=" << x<<" p= "<<p<<endl;
#endif  

    double a = p.perp()/pz;
    double b = x.perp()-a*x.z();
    
#ifdef MUPROPAGATOR_DEBUG  
  cout<<"MuPropagator::Propagate to disk="<<surface.position().z()<<" pz="<<pz<<
  " line parameters="<<a<<" "<<b<<endl;
#endif 
 
// Propagate on surface:phidet
//
  dfcalc = aCharge*0.006*abs(x.z()-surface.position().z())/abs(pz);
  phnext = phiold+dfcalc;
  
  if(phnext>twopi) phnext = phnext-twopi;
  if(phnext<0.) phnext = phnext+twopi;
  
// Propagate Zdet
//
  rdet = a*surface.position().z()-b;
  
// New state
  GlobalPoint xnew(rdet*cos(phnext),rdet*sin(phnext),surface.position().z());
  GlobalVector pnew(px*cos(dfcalc)-py*sin(dfcalc),px*sin(dfcalc)+py*cos(dfcalc),pz);
  
#ifdef MUPROPAGATOR_DEBUG
  cout<< "MuPropagator::phiold,phnext,zdet,charge,dfcalc="
  <<phiold<<" "<<phnext<<" "<<
  surface.position().z()<<" "<<aCharge<<" "<<dfcalc<<endl;
  cout<<"New coordinates="<<xnew<<endl;
  cout<<"New momentum="<<pnew<<endl;
#endif  

  TrajectoryStateOnSurface tsos(
                                 GlobalTrajectoryParameters(xnew, pnew, (int)aCharge, field),
                                 CurvilinearTrajectoryError(e),
			         surface);
     return tsos;
  }

}
