// ------------------------- //
// GEMSegFit.cc 
// Created:  11.05.2015
// Based on CSCSegFit.cc
// ------------------------- //

#include "RecoLocalMuon/GEMSegment/plugins/GEMSegFit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


bool GEMSegFit::fit(void) {
  if ( fitdone() ) return fitdone_; // don't redo fit unnecessarily
  short n = nhits();
  switch ( n ) {
  case 1:
    edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::fit] - cannot fit just 1 hit!!";
    break;
  case 2:
    fit2();
    break;
  case 3:
  case 4:
  case 5:
  case 6:
    fitlsq();
    break;
  default:
    edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::fit] - cannot fit more than 6 hits!!"; // should make this :: "cannot fit more than 4 hits!!"
  }  
  return fitdone_;
}

void GEMSegFit::fit2(void) {

  // Just join the two points
  // Equation of straight line between (x1, y1) and (x2, y2) in xy-plane is
  //       y = mx + c
  // with m = (y2-y1)/(x2-x1)
  // and  c = (y1*x2-x2*y1)/(x2-x1)
  //
  // Now we will make two straight lines
  // one in xz-plane, another in yz-plane
  //       x = uz + c1 
  //       y = vz + c2

  edm::LogVerbatim("GEMSegFit") << "[GEMSegFit:fit2]-------------------------------------";

  // 1) Check whether hits are on the same layer
  // -------------------------------------------
  GEMSetOfHits::const_iterator ih = hits_.begin();

  GEMDetId d1 = DetId((*ih)->rawId());
  int il1     = d1.layer();
  const GEMRecHit& h1 = (**ih);
  ++ih;    
  GEMDetId d2 = DetId((*ih)->rawId());
  int il2     = d2.layer();
  const GEMRecHit& h2 = (**ih);
    
  // Skip if on same layer, but should have been avoided earlier on 
  // (in SegAlgo.cc where clustering has been performed)
  if (il1 == il2) {
    edm::LogVerbatim("GEMSegFit") << "[GEMSegFit:fit2] - 2 hits on same layer!!";
    return;
  }
    

  // 2) Global Positions of hit 1 and 2 and
  //    Local  Positions of hit 1 and 2 w.r.t. reference GEM Eta Partition 
  // ---------------------------------------------------------------------
  const GEMEtaPartition* roll1 = gemetapartition(d1);
  GlobalPoint h1glopos = roll1->toGlobal(h1.localPosition());
  const GEMEtaPartition* roll2 = gemetapartition(d2);
  GlobalPoint h2glopos = roll2->toGlobal(h2.localPosition());
    
  // We want hit wrt first gem eta partition 
  // ( = reference m00 eta partition) 
  // (and local z will be != 0)
  LocalPoint h1pos = gemchamber()->toLocal(h1glopos);  
  LocalPoint h2pos = gemchamber()->toLocal(h2glopos);  
    

  // 3) Now make straight line between the two points in local coords
  // ----------------------------------------------------------------
  float dz = h2pos.z()-h1pos.z();
  if(dz != 0.0) {
    uslope_ = ( h2pos.x() - h1pos.x() ) / dz ;
    vslope_ = ( h2pos.y() - h1pos.y() ) / dz ;
  }

  float uintercept = ( h1pos.x()*h2pos.z() - h2pos.x()*h1pos.z() ) / dz;
  float vintercept = ( h1pos.y()*h2pos.z() - h2pos.y()*h1pos.z() ) / dz;

  // calculate local position (variable: intercept_)
  intercept_ = LocalPoint( uintercept, vintercept, 0.);

  // calculate the local direction (variable: localdir_)
  setOutFromIP();

  //@@ NOT SURE WHAT IS SENSIBLE FOR THESE...
  chi2_ = 0.;
  ndof_ = 0;

  fitdone_ = true;
  edm::LogVerbatim("GEMSegFit") << "[GEMSegFit:fit2]-------------------------------------";
  edm::LogVerbatim("GEMSegFit") << "\n\n";
}


void GEMSegFit::fitlsq(void) {

  edm::LogVerbatim("GEMSegFit") << "[GEMSegFit:fitlsq]-----------------------------------";  
  // Linear least-squares fit to up to 6 GEM rechits, one per layer in a GEM chamber.
  // Comments adapted from Tim Cox' comments in the original  GEMSegAlgoSK algorithm.
  
  // Fit to the local x, y rechit coordinates in z projection
  // The strip measurement controls the precision of x
  // The wire measurement controls the precision of y.
  // Typical precision: u (strip, sigma~200um), v (wire, sigma~1cm)
  
  // Set up the normal equations for the least-squares fit as a matrix equation
  
  // We have a vector of measurements m, which is a 2n x 1 dim matrix
  // The transpose mT is (u1, v1, u2, v2, ..., un, vn) where
  // ui is the strip-associated measurement and 
  // vi is the wire-associated measurement 
  // for a given rechit i.
  
  // The fit is to
  // u = u0 + uz * z
  // v = v0 + vz * z
  // where u0, uz, v0, vz are the parameters to be obtained from the fit.
  
  // These are contained in a vector p which is a 4x1 dim matrix, and
  // its transpose pT is (u0, v0, uz, vz). Note the ordering!
  
  // The covariance matrix for each pair of measurements is 2 x 2 and
  // the inverse of this is the error matrix E.
  // The error matrix for the whole set of n measurements is a diagonal
  // matrix with diagonal elements the individual 2 x 2 error matrices
  // (because the inverse of a diagonal matrix is a diagonal matrix
  // with each element the inverse of the original.)
  
  // In function 'weightMatrix()', the variable 'matrix' is filled with this
  // block-diagonal overall covariance matrix. Then 'matrix' is inverted to the 
  // block-diagonal error matrix, and returned.
  
  // Define the matrix A as
  //    1   0   z1  0
  //    0   1   0   z1
  //    1   0   z2  0
  //    0   1   0   z2
  //    ..  ..  ..  ..
  //    1   0   zn  0
  //    0   1   0   zn
  
  // This matrix A is set up and returned by function 'derivativeMatrix()'.
  
  // Then the normal equations are described by the matrix equation
  //
  //    (AT E A)p = (AT E)m
  //
  // where AT is the transpose of A.
  
  // Call the combined matrix on the LHS, M, and that on the RHS, B:
  //     M p = B
  
  // We solve this for the parameter vector, p.
  // The elements of M and B then involve sums over the hits
  
  // The covariance matrix of the parameters is obtained by 
  // (AT E A)^-1 calculated in 'covarianceMatrix()'.
  

  // NOTE
  // We need local position of a RecHit w.r.t. the CHAMBER
  // and the RecHit itself only knows its local position w.r.t.
  // the LAYER, so we must explicitly transform global position.
  

  SMatrix4 M; // 4x4, init to 0
  SVector4 B; // 4x1, init to 0; 

  GEMSetOfHits::const_iterator ih = hits_.begin();
  
  // LogDebug :: Loop over the TrackingRecHits and print the GEM Hits  
  // We don't need this ...
  /*
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) 
    {
      edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::fitlsq] - looping over GEMRecHits";
      const GEMRecHit& hit = (**ih);
      GEMDetId d = GEMDetId(hit.rawId());
      edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::fitlsq] - Tracking RecHit in detid ("<<d.rawId()<<")";
      edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::fitlsq] - GEMDetId ("<<GEMDetId(d.rawId())<<")";
    }
  */

  // Loop over the GEMRecHits and make small (2x2) matrices used to fill the blockdiagonal covariance matrix E^-1
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {
    const GEMRecHit& hit = (**ih);
    GEMDetId d = DetId(hit.rawId());
    const GEMEtaPartition* roll = gemetapartition(d);
    GlobalPoint gp         = roll->toGlobal(hit.localPosition());
    LocalPoint  lp         = gemchamber()->toLocal(gp); 
    
    // LogDebug
    #ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode
    std::stringstream lpss; lpss<<lp; std::string lps = lpss.str();
    std::stringstream gpss; gpss<<gp; std::string gps = gpss.str();
    edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::fitlsq] - Tracking RH glob pos "<<std::setw(35)<<gps<<" and loc pos "<<std::setw(35)<<lps
				  <<" wrt ref GEM chamber "<<gemchamber()->id().rawId()<<" = "<<gemchamber()->id();
    #endif

    // Local position of hit w.r.t. chamber
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    // Covariance matrix of local errors 
    SMatrixSym2 IC; // 2x2, init to 0
    
    IC(0,0) = hit.localPositionError().xx();
    IC(1,1) = hit.localPositionError().yy();
    //@@ NOT SURE WHICH OFF-DIAGONAL ELEMENT MUST BE DEFINED BUT (1,0) WORKS
    //@@ (and SMatrix enforces symmetry)
    IC(1,0) = hit.localPositionError().xy();
    // IC(0,1) = IC(1,0);
    
    edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::fit] 2x2 covariance matrix for this GEMRecHit :: [[" << IC(0,0) <<", "<< IC(0,1) <<"]["<< IC(1,0) <<","<<IC(1,1)<<"]]";

    // Invert covariance matrix (and trap if it fails!)
    bool ok = IC.Invert();
    if ( !ok ) {
      edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::fit] Failed to invert covariance matrix: \n" << IC;      
      //      return ok;  //@@ SHOULD PASS THIS BACK TO CALLER?
    }

    M(0,0) += IC(0,0);
    M(0,1) += IC(0,1);
    M(0,2) += IC(0,0) * z;
    M(0,3) += IC(0,1) * z;
    B(0)   += u * IC(0,0) + v * IC(0,1);
 
    M(1,0) += IC(1,0);
    M(1,1) += IC(1,1);
    M(1,2) += IC(1,0) * z;
    M(1,3) += IC(1,1) * z;
    B(1)   += u * IC(1,0) + v * IC(1,1);
 
    M(2,0) += IC(0,0) * z;
    M(2,1) += IC(0,1) * z;
    M(2,2) += IC(0,0) * z * z;
    M(2,3) += IC(0,1) * z * z;
    B(2)   += ( u * IC(0,0) + v * IC(0,1) ) * z;
 
    M(3,0) += IC(1,0) * z;
    M(3,1) += IC(1,1) * z;
    M(3,2) += IC(1,0) * z * z;
    M(3,3) += IC(1,1) * z * z;
    B(3)   += ( u * IC(1,0) + v * IC(1,1) ) * z;

  }

  SVector4 p;
  bool ok = M.Invert();
  if (!ok ){
    edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::fit] Failed to invert matrix: \n" << M;
    //    return ok; //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  else {
    p = M * B;
  }

  LogTrace("GEMSegFitMatrixDetails") << "[GEMSegFit::fit] p = " 
        << p(0) << ", " << p(1) << ", " << p(2) << ", " << p(3);
  
  // fill member variables  (note origin has local z = 0)
  //  intercept_
  intercept_ = LocalPoint(p(0), p(1), 0.);
  
  // localdir_ - set so segment points outwards from IP
  uslope_ = p(2);
  vslope_ = p(3);
  setOutFromIP();
  
  // calculate chi2 of fit
  setChi2( );

  // flag fit has been done
  fitdone_ = true;
  edm::LogVerbatim("GEMSegFit") << "[GEMSegFit:fitlsq]-----------------------------------";  
  edm::LogVerbatim("GEMSegFit") << "\n\n";  
}



void GEMSegFit::setChi2(void) {
  
  double chsq = 0.;

  GEMSetOfHits::const_iterator ih;

  // LogDebug
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {
    const GEMRecHit& hit = (**ih);
    GEMDetId d = GEMDetId(hit.rawId());
    const GEMEtaPartition* roll = gemetapartition(d);
    GlobalPoint gp         = roll->toGlobal(hit.localPosition());
    LocalPoint lp          = gemchamber()->toLocal(gp);    
    edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::setChi2] Local Point in GEMSuperChamber :: x, y, z = " << std::setw(12)<< lp.x() << ", " << std::setw(12)<< lp.y() << ", " << std::setw(12)<< lp.z() 
				  <<" [GEM RecHit in St "<<d.station()<<" La "<<d.layer()<<"]";
  }
  edm::LogVerbatim("GEMSegFit") << "-----------------------------------------------------";  

  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {

    const GEMRecHit& hit = (**ih);
    GEMDetId d = GEMDetId(hit.rawId());
    const GEMEtaPartition* roll = gemetapartition(d);
    GlobalPoint gp         = roll->toGlobal(hit.localPosition());
    LocalPoint lp          = gemchamber()->toLocal(gp);
    
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    double du = intercept_.x() + uslope_ * z - u;
    double dv = intercept_.y() + vslope_ * z - v;
    
    edm::LogVerbatim("GEMSegFit") << "[GEM RecHit in St "<<d.station()<<" La "<<d.layer()<<"] du = "<<intercept_.x()<<" + "<<uslope_<<" * "<<z<<" - "<<u<<" = "<<du;
    edm::LogVerbatim("GEMSegFit") << "[GEM RecHit in St "<<d.station()<<" La "<<d.layer()<<"] dv = "<<intercept_.y()<<" + "<<vslope_<<" * "<<z<<" - "<<v<<" = "<<dv;

    SMatrixSym2 IC; // 2x2, init to 0

    IC(0,0) = hit.localPositionError().xx();
    //    IC(0,1) = hit.localPositionError().xy();
    IC(1,0) = hit.localPositionError().xy();
    IC(1,1) = hit.localPositionError().yy();
    //    IC(1,0) = IC(0,1);

    edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::setChi2] IC before = \n" << IC;

    // Invert covariance matrix
    bool ok = IC.Invert();
    if (!ok ){
      edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::setChi2] Failed to invert covariance matrix: \n" << IC;
      //      return ok;
    }
    edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::setChi2] IC after = \n" << IC;
    chsq += du*du*IC(0,0) + 2.*du*dv*IC(0,1) + dv*dv*IC(1,1);
    edm::LogVerbatim("GEMSegFit") << "[GEM RecHit in St "<<d.station()<<" La "<<d.layer()<<"] Contribution to Chi^2 of this hit :: du^2*Cov(0,0) + 2*du*dv*Cov(0,1) + dv^2*IC(1,1) = "
				  <<du*du<<"*"<<IC(0,0)<<" + 2.*"<<du<<"*"<<dv<<"*"<<IC(0,1)<<" + "<<dv*dv<<"*"<<IC(1,1)<<" = "<<chsq;
  }
  
  // fill member variables
  chi2_ = chsq;
  ndof_ = 2.*hits_.size() - 4;

  edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::setChi2] chi2/ndof = " << chi2_ << "/" << ndof_ ;
  edm::LogVerbatim("GEMSegFit") << "-----------------------------------------------------";  

  // check fit quality ... maybe write a separate function later on
  // that is there only for debugging issues


  edm::LogVerbatim("GEMSegFit") << "[GEM Segment with Local Direction = "<<localdir_<<" and Local Position "<<intercept_<<"] can be written as:";
  edm::LogVerbatim("GEMSegFit") << "[ x ] = "<<localdir_.x()<<" * t + "<<intercept_.x();
  edm::LogVerbatim("GEMSegFit") << "[ y ] = "<<localdir_.y()<<" * t + "<<intercept_.y();
  edm::LogVerbatim("GEMSegFit") << "[ z ] = "<<localdir_.z()<<" * t + "<<intercept_.z();
  edm::LogVerbatim("GEMSegFit") << "Now extrapolate to each of the GEMRecHits XY plane (so constant z = RH LP.z()) to obtain [x1,y1]";

  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {

    const GEMRecHit& hit = (**ih);
    GEMDetId d = GEMDetId(hit.rawId());
    const GEMEtaPartition* roll = gemetapartition(d);
    GlobalPoint gp         = roll->toGlobal(hit.localPosition());
    LocalPoint lp          = gemchamber()->toLocal(gp);

    edm::LogVerbatim("GEMSegFit") << "[GEM RecHit in St "<<d.station()<<" La "<<d.layer()<<"] :: x, y, z = " << std::setw(12)<< lp.x() << ", " << std::setw(12)<< lp.y() << ", " << std::setw(12)<< lp.z();
    edm::LogVerbatim("GEMSegFit") << "[GEM RecHit in St "<<d.station()<<" La "<<d.layer()<<"] :: extrapolate the segment to xy-plane with z = "<<lp.z()<<" ==> param t = "<<(lp.z() - intercept_.z())/localdir_.z();
    double xtrap_z = lp.z();
    double xtrap_x = localdir_.x()*lp.z()/localdir_.z()+intercept_.x();
    double xtrap_y = localdir_.y()*lp.z()/localdir_.z()+intercept_.y();
    edm::LogVerbatim("GEMSegFit") << "[GEM RecHit in St "<<d.station()<<" La "<<d.layer()<<"] :: a, b, c = "<<std::setw(12)<<xtrap_x<<", "<<std::setw(12)<<xtrap_y<<", "<<std::setw(12)<<xtrap_z;
    edm::LogVerbatim("GEMSegFit") << "[GEM RecHit in St "<<d.station()<<" La "<<d.layer()<<"] :: delta x = "<<std::setw(12)<<(xtrap_x-lp.x())<<" delta y = "<<std::setw(12)<<(xtrap_y-lp.y());
    edm::LogVerbatim("GEMSegFit") << "" ;
  }




}




GEMSegFit::SMatrixSym12 GEMSegFit::weightMatrix() {
  
  bool ok = true;

  SMatrixSym12 matrix = ROOT::Math::SMatrixIdentity(); // 12x12, init to 1's on diag

  int row = 0;
  
  for (GEMSetOfHits::const_iterator it = hits_.begin(); it != hits_.end(); ++it) {
    
    const GEMRecHit& hit = (**it);

    // Note scaleXError allows rescaling the x error if necessary

    matrix(row, row)   = scaleXError()*hit.localPositionError().xx();
    matrix(row, row+1) = hit.localPositionError().xy();
    ++row;
    matrix(row, row-1) = hit.localPositionError().xy();
    matrix(row, row)   = hit.localPositionError().yy();
    ++row;
  }

  ok = matrix.Invert(); // invert in place
  if ( !ok ) {
    edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::weightMatrix] Failed to invert matrix: \n" << matrix;      
    //    return ok; //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  return matrix;
}




GEMSegFit::SMatrix12by4 GEMSegFit::derivativeMatrix() {
  
  SMatrix12by4 matrix; // 12x4, init to 0
  int row = 0;
  
  for( GEMSetOfHits::const_iterator it = hits_.begin(); it != hits_.end(); ++it) {
    
    const GEMRecHit& hit = (**it);
    GEMDetId d = GEMDetId(hit.rawId());
    const GEMEtaPartition* roll = gemetapartition(d);
    GlobalPoint gp = roll->toGlobal(hit.localPosition());
    LocalPoint lp = gemchamber()->toLocal(gp); 
    float z = lp.z();

    matrix(row, 0) = 1.;
    matrix(row, 2) = z;
    ++row;
    matrix(row, 1) = 1.;
    matrix(row, 3) = z;
    ++row;
  }
  return matrix;
}


void GEMSegFit::setOutFromIP() {
  // Set direction of segment to point from IP outwards
  // (Incorrect for particles not coming from IP, of course.)

  double dxdz = uslope_;
  double dydz = vslope_;
  double dz   = 1./sqrt(1. + dxdz*dxdz + dydz*dydz);
  double dx   = dz*dxdz;
  double dy   = dz*dydz;
  LocalVector localDir(dx,dy,dz);
  
  edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::setOutFromIP] :: dxdz = uslope_ = "<<std::setw(9)<<uslope_<<" dydz = vslope_ = "<<std::setw(9)<<vslope_<<" local dir = "<<localDir;

  // localDir sometimes needs a sign flip 
  // Examine its direction and origin in global z: to point outward
  // the localDir should always have same sign as global z...
  
  double globalZpos    = ( gemchamber()->toGlobal( intercept_ ) ).z();
  double globalZdir    = ( gemchamber()->toGlobal( localDir  ) ).z();
  localdir_ = ( localDir ).unit();

  edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::setOutFromIP] :: globalZpos = "<<globalZpos<<" globalZdir = "<<globalZdir<<" [sign should be the same]";
  edm::LogVerbatim("GEMSegFit") << "[GEMSegFit::setOutFromIP] ::  ==> local dir = "<<localdir_<< " localdir.phi = "<<localdir_.phi();
}



AlgebraicSymMatrix GEMSegFit::covarianceMatrix() {
  
  SMatrixSym12 weights = weightMatrix();
  SMatrix12by4 A = derivativeMatrix();
  edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::covarianceMatrix] weights matrix W: \n" << weights;      
  edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::covarianceMatrix] derivatives matrix A: \n" << A;      

  // (AT W A)^-1
  // e.g. See http://www.phys.ufl.edu/~avery/fitting.html, part I

  bool ok;
  SMatrixSym4 result =  ROOT::Math::SimilarityT(A, weights);
  edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::covarianceMatrix] (AT W A): \n" << result;      
  ok = result.Invert(); // inverts in place
  if ( !ok ) {
    edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::calculateError] Failed to invert matrix: \n" << result;      
    //    return ok;  //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::covarianceMatrix] (AT W A)^-1: \n" << result;      
  
  // reorder components to match TrackingRecHit interface (GEMSegment isa TrackingRecHit)
  // i.e. slopes first, then positions 
  AlgebraicSymMatrix flipped = flipErrors( result );
    
  return flipped;
}


AlgebraicSymMatrix GEMSegFit::flipErrors( const SMatrixSym4& a ) { 
    
  // The GEMSegment needs the error matrix re-arranged to match
  // parameters in order (uz, vz, u0, v0) 
  // where uz, vz = slopes, u0, v0 = intercepts
    
  edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::flipErrors] input: \n" << a;      

  AlgebraicSymMatrix hold(4, 0. ); 
      
  for ( short j=0; j!=4; ++j) {
    for (short i=0; i!=4; ++i) {
      hold(i+1,j+1) = a(i,j); // SMatrix counts from 0, AlgebraicMatrix from 1
    }
  }

  edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::flipErrors] after copy:";
  edm::LogVerbatim("GEMSegFitMatrixDetails") << "(" << hold(1,1) << "  " << hold(1,2) << "  " << hold(1,3) << "  " << hold(1,4);
  edm::LogVerbatim("GEMSegFitMatrixDetails") << " " << hold(2,1) << "  " << hold(2,2) << "  " << hold(2,3) << "  " << hold(2,4);
  edm::LogVerbatim("GEMSegFitMatrixDetails") << " " << hold(3,1) << "  " << hold(3,2) << "  " << hold(3,3) << "  " << hold(3,4);
  edm::LogVerbatim("GEMSegFitMatrixDetails") << " " << hold(4,1) << "  " << hold(4,2) << "  " << hold(4,3) << "  " << hold(4,4) << ")";

  // errors on slopes into upper left 
  hold(1,1) = a(2,2); 
  hold(1,2) = a(2,3); 
  hold(2,1) = a(3,2); 
  hold(2,2) = a(3,3); 
    
  // errors on positions into lower right 
  hold(3,3) = a(0,0); 
  hold(3,4) = a(0,1); 
  hold(4,3) = a(1,0); 
  hold(4,4) = a(1,1); 
    
  // must also interchange off-diagonal elements of off-diagonal 2x2 submatrices
  hold(4,1) = a(1,2);
  hold(3,2) = a(0,3);
  hold(2,3) = a(3,0); // = a(0,3)
  hold(1,4) = a(2,1); // = a(1,2)

  edm::LogVerbatim("GEMSegFitMatrixDetails") << "[GEMSegFit::flipErrors] after flip:";
  edm::LogVerbatim("GEMSegFitMatrixDetails") << "(" << hold(1,1) << "  " << hold(1,2) << "  " << hold(1,3) << "  " << hold(1,4);
  edm::LogVerbatim("GEMSegFitMatrixDetails") << " " << hold(2,1) << "  " << hold(2,2) << "  " << hold(2,3) << "  " << hold(2,4);
  edm::LogVerbatim("GEMSegFitMatrixDetails") << " " << hold(3,1) << "  " << hold(3,2) << "  " << hold(3,3) << "  " << hold(3,4);
  edm::LogVerbatim("GEMSegFitMatrixDetails") << " " << hold(4,1) << "  " << hold(4,2) << "  " << hold(4,3) << "  " << hold(4,4) << ")";

  return hold;
}
 
float GEMSegFit::xfit( float z ) const {
  //@@ ADD THIS TO EACH ACCESSOR OF FIT RESULTS?
  //  if ( !fitdone() ) fit();
  return intercept_.x() + uslope_ * z;
}

float GEMSegFit::yfit( float z ) const {
  return intercept_.y() + vslope_ * z;
}

float GEMSegFit::xdev( float x, float z ) const {
  return intercept_.x() + uslope_ * z - x;
}

float GEMSegFit::ydev( float y, float z ) const {
  return intercept_.y() + vslope_ * z - y;
}

float GEMSegFit::Rdev(float x, float y, float z) const {
  return sqrt ( xdev(x,z)*xdev(x,z) + ydev(y,z)*ydev(y,z) );
}

