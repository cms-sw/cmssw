// ------------------------- //
// MuonSegFit.cc 
// Created:  11.05.2015
// Based on CSCSegFit.cc
// ------------------------- //

#include "RecoLocalMuon/GEMSegment/plugins/MuonSegFit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


bool MuonSegFit::fit(void) {
  if ( fitdone() ) return fitdone_; // don't redo fit unnecessarily
  short n = nhits();
  if (n < 2){
    edm::LogVerbatim("MuonSegFit") << "[MuonSegFit::fit] - cannot fit just 1 hit!!";
  }
  else if (n == 2){
    fit2();
  }
  else if (2*n <= MaxHits2){
    fitlsq();
  }
  else {
    edm::LogVerbatim("MuonSegFit") << "[MuonSegFit::fit] - cannot fit more than "<< MaxHits2/2 <<" hits!!";
  }
  return fitdone_;
}

void MuonSegFit::fit2(void) {
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
  // 1) Check whether hits are on the same layer
  // should be done before, so removed
  // -------------------------------------------

  MuonRecHitContainer::const_iterator ih = hits_.begin();
  // 2) Global Positions of hit 1 and 2 and
  //    Local  Positions of hit 1 and 2 w.r.t. reference GEM Eta Partition 
  // ---------------------------------------------------------------------
  LocalPoint h1pos = (*ih)->localPosition();
  ++ih;
  LocalPoint h2pos = (*ih)->localPosition();
  
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
}


void MuonSegFit::fitlsq(void) {
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

  SMatrix4 M; // 4x4, init to 0
  SVector4 B; // 4x1, init to 0; 

  MuonRecHitContainer::const_iterator ih = hits_.begin();
  
  // Loop over the GEMRecHits and make small (2x2) matrices used to fill the blockdiagonal covariance matrix E^-1
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {
    LocalPoint  lp         = (*ih)->localPosition();    
    // Local position of hit w.r.t. chamber
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();

    // Covariance matrix of local errors 
    SMatrixSym2 IC; // 2x2, init to 0
    
    IC(0,0) = (*ih)->localPositionError().xx();
    IC(1,1) = (*ih)->localPositionError().yy();
    //@@ NOT SURE WHICH OFF-DIAGONAL ELEMENT MUST BE DEFINED BUT (1,0) WORKS
    //@@ (and SMatrix enforces symmetry)
    IC(1,0) = (*ih)->localPositionError().xy();
    // IC(0,1) = IC(1,0);
    
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::fit] 2x2 covariance matrix for this GEMRecHit :: [[" << IC(0,0) <<", "<< IC(0,1) <<"]["<< IC(1,0) <<","<<IC(1,1)<<"]]";
#endif

    // Invert covariance matrix (and trap if it fails!)
    bool ok = IC.Invert();
    if ( !ok ) {
      edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::fit] Failed to invert covariance matrix: \n" << IC;      
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
    edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::fit] Failed to invert matrix: \n" << M;
    //    return ok; //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  else {
    p = M * B;
  }

#ifdef EDM_ML_DEBUG
  LogTrace("MuonSegFitMatrixDetails") << "[MuonSegFit::fit] p = " 
				      << p(0) << ", " << p(1) << ", " << p(2) << ", " << p(3);
#endif
  
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
}



void MuonSegFit::setChi2(void) {
  
  double chsq = 0.;

  MuonRecHitContainer::const_iterator ih;

  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {
    LocalPoint lp = (*ih)->localPosition();
    
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    double du = intercept_.x() + uslope_ * z - u;
    double dv = intercept_.y() + vslope_ * z - v;
    
    SMatrixSym2 IC; // 2x2, init to 0

    IC(0,0) = (*ih)->localPositionError().xx();
    //    IC(0,1) = (*ih)->localPositionError().xy();
    IC(1,0) = (*ih)->localPositionError().xy();
    IC(1,1) = (*ih)->localPositionError().yy();
    //    IC(1,0) = IC(0,1);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::setChi2] IC before = \n" << IC;
#endif

    // Invert covariance matrix
    bool ok = IC.Invert();
    if (!ok ){
      edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::setChi2] Failed to invert covariance matrix: \n" << IC;
      //      return ok;
    }
    chsq += du*du*IC(0,0) + 2.*du*dv*IC(0,1) + dv*dv*IC(1,1);
    
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::setChi2] IC after = \n" << IC;
    edm::LogVerbatim("MuonSegFit") << "[GEM RecHit ] Contribution to Chi^2 of this hit :: du^2*Cov(0,0) + 2*du*dv*Cov(0,1) + dv^2*IC(1,1) = "<<du*du<<"*"<<IC(0,0)<<" + 2.*"<<du<<"*"<<dv<<"*"<<IC(0,1)<<" + "<<dv*dv<<"*"<<IC(1,1)<<" = "<<chsq;
#endif    
  }
  
  // fill member variables
  chi2_ = chsq;
  ndof_ = 2.*hits_.size() - 4;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSegFit") << "[MuonSegFit::setChi2] chi2/ndof = " << chi2_ << "/" << ndof_ ;
  edm::LogVerbatim("MuonSegFit") << "-----------------------------------------------------";  

  // check fit quality ... maybe write a separate function later on
  // that is there only for debugging issues

  edm::LogVerbatim("MuonSegFit") << "[GEM Segment with Local Direction = "<<localdir_<<" and Local Position "<<intercept_<<"] can be written as:";
  edm::LogVerbatim("MuonSegFit") << "[ x ] = "<<localdir_.x()<<" * t + "<<intercept_.x();
  edm::LogVerbatim("MuonSegFit") << "[ y ] = "<<localdir_.y()<<" * t + "<<intercept_.y();
  edm::LogVerbatim("MuonSegFit") << "[ z ] = "<<localdir_.z()<<" * t + "<<intercept_.z();
  edm::LogVerbatim("MuonSegFit") << "Now extrapolate to each of the GEMRecHits XY plane (so constant z = RH LP.z()) to obtain [x1,y1]";
#endif
}

MuonSegFit::SMatrixSym12 MuonSegFit::weightMatrix() {
  
  bool ok = true;

  SMatrixSym12 matrix = ROOT::Math::SMatrixIdentity(); // 12x12, init to 1's on diag

  int row = 0;
  
  for (MuonRecHitContainer::const_iterator it = hits_.begin(); it != hits_.end(); ++it) {
    // Note scaleXError allows rescaling the x error if necessary
    matrix(row, row)   = scaleXError()*(*it)->localPositionError().xx();
    matrix(row, row+1) = (*it)->localPositionError().xy();
    ++row;
    matrix(row, row-1) = (*it)->localPositionError().xy();
    matrix(row, row)   = (*it)->localPositionError().yy();
    ++row;
  }

  ok = matrix.Invert(); // invert in place
  if ( !ok ) {
    edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::weightMatrix] Failed to invert matrix: \n" << matrix;      
    //    return ok; //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  return matrix;
}


MuonSegFit::SMatrix12by4 MuonSegFit::derivativeMatrix() {
  
  SMatrix12by4 matrix; // 12x4, init to 0
  int row = 0;
  
  for( MuonRecHitContainer::const_iterator it = hits_.begin(); it != hits_.end(); ++it) {
    LocalPoint  lp         = (*it)->localPosition();
    
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


void MuonSegFit::setOutFromIP() {
  // Set direction of segment to point from IP outwards
  // (Incorrect for particles not coming from IP, of course.)
  
  double dxdz = uslope_;
  double dydz = vslope_;
  double dz   = 1./sqrt(1. + dxdz*dxdz + dydz*dydz);
  double dx   = dz*dxdz;
  double dy   = dz*dydz;
  LocalVector localDir(dx,dy,dz);

  localdir_ = ( localDir ).unit();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSegFit") << "[MuonSegFit::setOutFromIP] :: dxdz = uslope_ = "<<std::setw(9)<<uslope_<<" dydz = vslope_ = "<<std::setw(9)<<vslope_<<" local dir = "<<localDir;
  edm::LogVerbatim("MuonSegFit") << "[MuonSegFit::setOutFromIP] ::  ==> local dir = "<<localdir_<< " localdir.phi = "<<localdir_.phi();
#endif
}



AlgebraicSymMatrix MuonSegFit::covarianceMatrix() {
  
  SMatrixSym12 weights = weightMatrix();
  SMatrix12by4 A = derivativeMatrix();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::covarianceMatrix] weights matrix W: \n" << weights;      
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::covarianceMatrix] derivatives matrix A: \n" << A;      
#endif

  // (AT W A)^-1
  // e.g. See http://www.phys.ufl.edu/~avery/fitting.html, part I

  bool ok;
  SMatrixSym4 result =  ROOT::Math::SimilarityT(A, weights);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::covarianceMatrix] (AT W A): \n" << result;
#endif
  
  ok = result.Invert(); // inverts in place
  if ( !ok ) {
    edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::calculateError] Failed to invert matrix: \n" << result;      
    //    return ok;  //@@ SHOULD PASS THIS BACK TO CALLER?
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::covarianceMatrix] (AT W A)^-1: \n" << result;      
#endif
  
  // reorder components to match TrackingRecHit interface (GEMSegment isa TrackingRecHit)
  // i.e. slopes first, then positions 
  AlgebraicSymMatrix flipped = flipErrors( result );
    
  return flipped;
}


AlgebraicSymMatrix MuonSegFit::flipErrors( const SMatrixSym4& a ) { 
    
  // The GEMSegment needs the error matrix re-arranged to match
  // parameters in order (uz, vz, u0, v0) 
  // where uz, vz = slopes, u0, v0 = intercepts
    
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::flipErrors] input: \n" << a;      
#endif

  AlgebraicSymMatrix hold(4, 0. ); 
      
  for ( short j=0; j!=4; ++j) {
    for (short i=0; i!=4; ++i) {
      hold(i+1,j+1) = a(i,j); // SMatrix counts from 0, AlgebraicMatrix from 1
    }
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::flipErrors] after copy:";
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "(" << hold(1,1) << "  " << hold(1,2) << "  " << hold(1,3) << "  " << hold(1,4);
  edm::LogVerbatim("MuonSegFitMatrixDetails") << " " << hold(2,1) << "  " << hold(2,2) << "  " << hold(2,3) << "  " << hold(2,4);
  edm::LogVerbatim("MuonSegFitMatrixDetails") << " " << hold(3,1) << "  " << hold(3,2) << "  " << hold(3,3) << "  " << hold(3,4);
  edm::LogVerbatim("MuonSegFitMatrixDetails") << " " << hold(4,1) << "  " << hold(4,2) << "  " << hold(4,3) << "  " << hold(4,4) << ")";
#endif

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

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "[MuonSegFit::flipErrors] after flip:";
  edm::LogVerbatim("MuonSegFitMatrixDetails") << "(" << hold(1,1) << "  " << hold(1,2) << "  " << hold(1,3) << "  " << hold(1,4);
  edm::LogVerbatim("MuonSegFitMatrixDetails") << " " << hold(2,1) << "  " << hold(2,2) << "  " << hold(2,3) << "  " << hold(2,4);
  edm::LogVerbatim("MuonSegFitMatrixDetails") << " " << hold(3,1) << "  " << hold(3,2) << "  " << hold(3,3) << "  " << hold(3,4);
  edm::LogVerbatim("MuonSegFitMatrixDetails") << " " << hold(4,1) << "  " << hold(4,2) << "  " << hold(4,3) << "  " << hold(4,4) << ")";
#endif

  return hold;
}
 
float MuonSegFit::xfit( float z ) const {
  //@@ ADD THIS TO EACH ACCESSOR OF FIT RESULTS?
  //  if ( !fitdone() ) fit();
  return intercept_.x() + uslope_ * z;
}

float MuonSegFit::yfit( float z ) const {
  return intercept_.y() + vslope_ * z;
}

float MuonSegFit::xdev( float x, float z ) const {
  return intercept_.x() + uslope_ * z - x;
}

float MuonSegFit::ydev( float y, float z ) const {
  return intercept_.y() + vslope_ * z - y;
}

float MuonSegFit::Rdev(float x, float y, float z) const {
  return sqrt ( xdev(x,z)*xdev(x,z) + ydev(y,z)*ydev(y,z) );
}

