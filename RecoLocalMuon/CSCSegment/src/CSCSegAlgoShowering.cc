/**
 * \file CSCSegAlgoShowering.cc
 *
 *  \author: D. Fortin - UC Riverside
 *
 * See header file for description.
 */

#include "RecoLocalMuon/CSCSegment/src/CSCSegAlgoShowering.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>


/* Constructor
 *
 */
CSCSegAlgoShowering::CSCSegAlgoShowering(const edm::ParameterSet& ps) {
  debug                  = ps.getUntrackedParameter<bool>("CSCSegmentDebug");
}


/* Destructor:
 *
 */
CSCSegAlgoShowering::~CSCSegAlgoShowering(){

}


/* clusterHits
 *
 */
CSCSegment CSCSegAlgoShowering::showerSeg( const CSCChamber* theChamber, ChamberHitContainer rechits ) {

  // Initialize parameters
  std::vector<float> x, y, gz;
  std::vector<int> n;
 
  for (int i = 0; i < 6; ++i) {
    x.push_back(0.);
    y.push_back(0.);
    gz.push_back(0.);
    n.push_back(0);
  }

  // Loop over hits to find center-of-mass position in each layer
  for (ChamberHitContainer::const_iterator it = rechits.begin(); it != rechits.end(); it++ ) {
    const CSCRecHit2D& hit = (**it);
    const CSCDetId id = hit.cscDetId();
    int l_id = id.layer();
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint  lp         = theChamber->toLocal(gp);

    n[l_id -1]++;
    x[l_id -1] += lp.x();
    y[l_id -1] += lp.y();
    gz[l_id -1] += gp.z();
  }


  std::vector<LocalPoint> lpCOM;
  LocalPoint test(9999.,9999.,0);
  // Determine center of mass for each layer  
  for (unsigned i = 0; i < 6; ++i) {
    if (n[i] > 0) {
      x[i]/n[i];
      y[i]/n[i];
      LocalPoint lpt(x[i],y[i],0.);
      lpCOM.push_back(lpt);
    }
    else {
      lpCOM.push_back(test);
    }
  }


  std::vector<float> r_closest;
  std::vector<int> id;
  for (unsigned i = 0; i < 6; ++i ) {
    id.push_back(-1);
    r_closest.push_back(9999.);
  }

  int idx = 0;

  // Loop over all hits and find hit closest to com for that layer.
  for (ChamberHitContainer::const_iterator it = rechits.begin(); it != rechits.end(); it++ ) {    
    const CSCRecHit2D& hit = (**it);
    int l_id = hit.cscDetId().layer();
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint  lp         = theChamber->toLocal(gp);

    float d_x = lp.x() -lpCOM[l_id -1].x();
    float d_y = lp.x() -lpCOM[l_id -1].x();

    LocalPoint diff(d_x, d_y, 0.);
    
    if ( fabs(diff.mag() ) < r_closest[l_id -1] ) {
       r_closest[l_id -1] =  fabs(diff.mag());
       id[l_id -1] = idx;
    }
    idx++;
    
  }

  // Now fill vector of rechits closest to center of mass:
  idx = 0;
  ChamberHitContainer protoSegment;

  // Loop over all hits and find hit closest to com for that layer.
  for (ChamberHitContainer::const_iterator it = rechits.begin(); it != rechits.end(); it++ ) {    
    const CSCRecHit2D& hit = (**it);
    int l_id = hit.cscDetId().layer();

    if ( idx == id[l_id -1] ) {
       protoSegment.push_back(*it);
    }
    idx++;
    
  }

  // Reorder hits in protosegment
  if ( gz[0] > 0. ) {
    if ( gz[0] > gz[5] ) { 
      reverse( protoSegment.begin(), protoSegment.end() );
    }    
  }
  else if ( gz[0] < 0. ) {
    if ( gz[0] < gz[5] ) {
      reverse( protoSegment.begin(), protoSegment.end() );
    }    
  }



  // Compute Segment slope and Intercept from Least Square Fit    

  float protoSlope_u = 0.;
  float protoSlope_v = 0.;
  double protoChi2 = 1.;
  LocalPoint  protoIntercept;

  HepMatrix M(4,4,0);
  HepVector B(4,0);

  ChamberHitContainer::const_iterator ih;
  
  for (ih = protoSegment.begin(); ih != protoSegment.end(); ++ih) {
    
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint  lp         = theChamber->toLocal(gp); 
    
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    // ptc: Covariance matrix of local errors 
    HepMatrix IC(2,2);
    IC(1,1) = hit.localPositionError().xx();
    IC(1,2) = hit.localPositionError().xy();
    IC(2,2) = hit.localPositionError().yy();
    IC(2,1) = IC(1,2); // since Cov is symmetric
    
    // ptc: Invert covariance matrix (and trap if it fails!)
    int ierr = 0;
    IC.invert(ierr); // inverts in place
    if (ierr != 0) {
      LogDebug("CSC") << "CSCSegment::fitSlopes: failed to invert covariance matrix=\n" << IC << "\n";      
    }
    
    M(1,1) += IC(1,1);
    M(1,2) += IC(1,2);
    M(1,3) += IC(1,1) * z;
    M(1,4) += IC(1,2) * z;
    B(1)   += u * IC(1,1) + v * IC(1,2);
    
    M(2,1) += IC(2,1);
    M(2,2) += IC(2,2);
    M(2,3) += IC(2,1) * z;
    M(2,4) += IC(2,2) * z;
    B(2)   += u * IC(2,1) + v * IC(2,2);
    
    M(3,1) += IC(1,1) * z;
    M(3,2) += IC(1,2) * z;
    M(3,3) += IC(1,1) * z * z;
    M(3,4) += IC(1,2) * z * z;
    B(3)   += ( u * IC(1,1) + v * IC(1,2) ) * z;
    
    M(4,1) += IC(2,1) * z;
    M(4,2) += IC(2,2) * z;
    M(4,3) += IC(2,1) * z * z;
    M(4,4) += IC(2,2) * z * z;
    B(4)   += ( u * IC(2,1) + v * IC(2,2) ) * z;
  }
  
  HepVector p = solve(M, B);
  
  // Update member variables 
  // Note that origin has local z = 0

  protoIntercept = LocalPoint(p(1), p(2), 0.);
  protoSlope_u = p(3);
  protoSlope_v = p(4);


  // Determine Chi^2 for the proto wire segment
  
  double chsq = 0.;
  
  for (ih = protoSegment.begin(); ih != protoSegment.end(); ++ih) {
    
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint lp          = theChamber->toLocal(gp);
    
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    double du = protoIntercept.x() + protoSlope_u * z - u;
    double dv = protoIntercept.y() + protoSlope_v * z - v;
    
    HepMatrix IC(2,2);
    IC(1,1) = hit.localPositionError().xx();
    IC(1,2) = hit.localPositionError().xy();
    IC(2,2) = hit.localPositionError().yy();
    IC(2,1) = IC(1,2);
    
    // Invert covariance matrix
    int ierr = 0;
    IC.invert(ierr);
    if (ierr != 0) {
      LogDebug("CSC") << "CSCSegment::fillChiSquared: failed to invert covariance matrix=\n" << IC << "\n";      
    }
    chsq += du*du*IC(1,1) + 2.*du*dv*IC(1,2) + dv*dv*IC(2,2);
  }

  protoChi2 = chsq;


  // Blightly assume the following never fails
 
  std::vector<const CSCRecHit2D*>::const_iterator it;
  int nhits = protoSegment.size();
  int ierr; 

  AlgebraicSymMatrix weights(2*nhits, 0);
  AlgebraicMatrix A(2*nhits, 4);

  int row = 0;  
  for (it = protoSegment.begin(); it != protoSegment.end(); ++it) {
    const CSCRecHit2D& hit = (**it);
    const CSCLayer* layer = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp = layer->toGlobal(hit.localPosition());      
    LocalPoint lp = theChamber->toLocal(gp); 
    float z = lp.z();
    ++row;
    weights(row, row)   = hit.localPositionError().xx();
    weights(row, row+1) = hit.localPositionError().xy();
    A(row, 1) = 1.;
    A(row, 3) = z;
    ++row;
    weights(row, row-1) = hit.localPositionError().xy();
    weights(row, row)   = hit.localPositionError().yy();
    A(row, 2) = 1.;
    A(row, 4) = z;
  }
  weights.invert(ierr);

  AlgebraicSymMatrix a = weights.similarityT(A);
  a.invert(ierr);
    
  // but reorder components to match what's required by TrackingRecHit interface 
  // i.e. slopes first, then positions 
    
  AlgebraicSymMatrix hold( a ); 
    
  // errors on slopes into upper left 
  a(1,1) = hold(3,3); 
  a(1,2) = hold(3,4); 
  a(2,1) = hold(4,3); 
  a(2,2) = hold(4,4); 
    
  // errors on positions into lower right 
  a(3,3) = hold(1,1); 
  a(3,4) = hold(1,2); 
  a(4,3) = hold(2,1); 
  a(4,4) = hold(2,2); 
    
  // off-diagonal elements remain unchanged such that 
  // Error matrix
  AlgebraicSymMatrix protoErrors = a;     


  // Form segment proper:

  // Local direction
  double dz   = 1./sqrt(1. + protoSlope_u*protoSlope_u + protoSlope_v*protoSlope_v);
  double dx   = dz*protoSlope_u;
  double dy   = dz*protoSlope_v;
  LocalVector localDir(dx,dy,dz);
        
  // localDir may need sign flip to ensure it points outward from IP  
  double globalZpos    = ( theChamber->toGlobal( protoIntercept ) ).z();
  double globalZdir    = ( theChamber->toGlobal( localDir ) ).z();
  double directionSign = globalZpos * globalZdir;
  LocalVector protoDirection = (directionSign * localDir).unit();
        
  CSCSegment temp(protoSegment, protoIntercept, protoDirection, protoErrors, protoChi2); 

  return temp;

} 


