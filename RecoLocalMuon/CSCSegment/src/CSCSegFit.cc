// CSCSegFit.cc 
// Last mod: 03.02.2015

#include "RecoLocalMuon/CSCSegment/src/CSCSegFit.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void CSCSegFit::fit(void) {
  if ( fitdone() ) return; // don't redo fit unnecessarily
  short n = nhits();
  switch ( n ) {
  case 1:
    edm::LogVerbatim("CSCSegFit") << "[CSCSegFit::fit] - cannot fit just 1 hit!!";
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
    edm::LogVerbatim("CSCSegFit") << "[CSCSegFit::fit] - cannot fit more than 6 hits!!";
  }  
}

void CSCSegFit::fit2(void) {

  // Just join the two points
  // Equation of straight line between (x1, y1) and (x2, y2) in xy-plane is
  //       y = mx + c
  // with m = (y2-y1)/(x2-x1)
  // and  c = (y1*x2-x2*y1)/(x2-x1)

  CSCSetOfHits::const_iterator ih = hits_.begin();
  int il1 = (*ih)->cscDetId().layer();
  const CSCRecHit2D& h1 = (**ih);
  ++ih;    
  int il2 = (*ih)->cscDetId().layer();
  const CSCRecHit2D& h2 = (**ih);
    
  // Skip if on same layer, but should be impossible :)
  if (il1 == il2) {
    edm::LogVerbatim("CSCSegFit") << "[CSCSegFit:fit]2 - 2 hits on same layer!!";
    return;
  }
    
  const CSCLayer* layer1 = chamber()->layer(il1);
  const CSCLayer* layer2 = chamber()->layer(il2);
    
  GlobalPoint h1glopos = layer1->toGlobal(h1.localPosition());
  GlobalPoint h2glopos = layer2->toGlobal(h2.localPosition());
    
  // We want hit wrt chamber (and local z will be != 0)
  LocalPoint h1pos = chamber()->toLocal(h1glopos);  
  LocalPoint h2pos = chamber()->toLocal(h2glopos);  
    
  float dz = h2pos.z()-h1pos.z();

  uslope_ = ( h2pos.x() - h1pos.x() ) / dz ;
  vslope_ = ( h2pos.y() - h1pos.y() ) / dz ;

  float uintercept = ( h1pos.x()*h2pos.z() - h2pos.x()*h1pos.z() ) / dz;
  float vintercept = ( h1pos.y()*h2pos.z() - h2pos.y()*h1pos.z() ) / dz;
  intercept_ = LocalPoint( uintercept, vintercept, 0.);

  setOutFromIP();

  //@@ NOT SURE WHAT IS SENSIBLE FOR THESE...
  chi2_ = 0.;
  ndof_ = 0;

  fitdone_ = true;
}


void CSCSegFit::fitlsq(void) {
  
  // Linear least-squares fit to up to 6 CSC rechits, one per layer in a CSC.
  // Comments adapted from  mine in original  CSCSegAlgoSK algorithm.
  
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

  CSCSetOfHits::const_iterator ih = hits_.begin();
  
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer  = chamber()->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint  lp         = chamber()->toLocal(gp); 
    
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
    
    // Invert covariance matrix (and trap if it fails!)
    bool ok = IC.Invert();
    if ( !ok ) {
      edm::LogVerbatim("CSCSegment|CSCSegFit") << "[CSCSegFit::fit] Failed to invert covariance matrix: \n" << IC;      
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
    edm::LogVerbatim("CSCSegment|CSCSegFit") << "[CSCSegFit::fit] Failed to invert matrix: \n" << M;
    //    return ok; //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  else {
    p = M * B;
  }

  //  LogTrace("CSCSegFit") << "[CSCSegFit::fit] p = " 
  //        << p(0) << ", " << p(1) << ", " << p(2) << ", " << p(3);
  
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



void CSCSegFit::setChi2(void) {
  
  double chsq = 0.;

  CSCSetOfHits::const_iterator ih;
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {

    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer  = chamber()->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint lp          = chamber()->toLocal(gp);
    
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    double du = intercept_.x() + uslope_ * z - u;
    double dv = intercept_.y() + vslope_ * z - v;
    
    //    LogTrace("CSCSegFit") << "[CSCSegFit::setChi2] u, v, z = " << u << ", " << v << ", " << z;

    SMatrixSym2 IC; // 2x2, init to 0

    IC(0,0) = hit.localPositionError().xx();
    //    IC(0,1) = hit.localPositionError().xy();
    IC(1,0) = hit.localPositionError().xy();
    IC(1,1) = hit.localPositionError().yy();
    //    IC(1,0) = IC(0,1);

    //    LogTrace("CSCSegFit") << "[CSCSegFit::setChi2] IC before = \n" << IC;

    // Invert covariance matrix
    bool ok = IC.Invert();
    if (!ok ){
      edm::LogVerbatim("CSCSegment|CSCSegFit") << "[CSCSegFit::setChi2] Failed to invert covariance matrix: \n" << IC;
      //      return ok;
    }
    //    LogTrace("CSCSegFit") << "[CSCSegFit::setChi2] IC after = \n" << IC;
    chsq += du*du*IC(0,0) + 2.*du*dv*IC(0,1) + dv*dv*IC(1,1);
  }
  
  // fill member variables
  chi2_ = chsq;
  ndof_ = 2.*hits_.size() - 4;

  //  LogTrace("CSCSegFit") << "[CSCSegFit::setChi2] chi2 = " << chi2_ << "/" << ndof_ << " dof";

}




CSCSegFit::SMatrixSym12 CSCSegFit::weightMatrix() {
  
  bool ok = true;

  SMatrixSym12 matrix = ROOT::Math::SMatrixIdentity(); // 12x12, init to 1's on diag

  int row = 0;
  
  for (CSCSetOfHits::const_iterator it = hits_.begin(); it != hits_.end(); ++it) {
    
    const CSCRecHit2D& hit = (**it);

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
    edm::LogVerbatim("CSCSegment|CSCSegFit") << "[CSCSegFit::weightMatrix] Failed to invert matrix: \n" << matrix;      
    //    return ok; //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  return matrix;
}




CSCSegFit::SMatrix12by4 CSCSegFit::derivativeMatrix() {
  
  SMatrix12by4 matrix; // 12x4, init to 0
  int row = 0;
  
  for( CSCSetOfHits::const_iterator it = hits_.begin(); it != hits_.end(); ++it) {
    
    const CSCRecHit2D& hit = (**it);
    const CSCLayer* layer = chamber()->layer(hit.cscDetId().layer());
    GlobalPoint gp = layer->toGlobal(hit.localPosition());      
    LocalPoint lp = chamber()->toLocal(gp); 
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


void CSCSegFit::setOutFromIP() {
  // Set direction of segment to point from IP outwards
  // (Incorrect for particles not coming from IP, of course.)
  
  double dxdz = uslope_;
  double dydz = vslope_;
  double dz   = 1./sqrt(1. + dxdz*dxdz + dydz*dydz);
  double dx   = dz*dxdz;
  double dy   = dz*dydz;
  LocalVector localDir(dx,dy,dz);

  // localDir sometimes needs a sign flip 
  // Examine its direction and origin in global z: to point outward
  // the localDir should always have same sign as global z...
  
  double globalZpos    = ( chamber()->toGlobal( intercept_ ) ).z();
  double globalZdir    = ( chamber()->toGlobal( localDir  ) ).z();
  double directionSign = globalZpos * globalZdir;
  localdir_ = (directionSign * localDir ).unit();
}



AlgebraicSymMatrix CSCSegFit::covarianceMatrix() {
  
  SMatrixSym12 weights = weightMatrix();
  SMatrix12by4 A = derivativeMatrix();
  //  LogTrace("CSCSegFit") << "[CSCSegFit::covarianceMatrix] weights matrix W: \n" << weights;      
  //  LogTrace("CSCSegFit") << "[CSCSegFit::covarianceMatrix] derivatives matrix A: \n" << A;      

  // (AT W A)^-1
  // e.g. See http://www.phys.ufl.edu/~avery/fitting.html, part I

  bool ok;
  SMatrixSym4 result =  ROOT::Math::SimilarityT(A, weights);
  //  LogTrace("CSCSegFit") << "[CSCSegFit::covarianceMatrix] (AT W A): \n" << result;      
  ok = result.Invert(); // inverts in place
  if ( !ok ) {
    edm::LogVerbatim("CSCSegment|CSCSegFit") << "[CSCSegFit::calculateError] Failed to invert matrix: \n" << result;      
    //    return ok;  //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  //  LogTrace("CSCSegFit") << "[CSCSegFit::covarianceMatrix] (AT W A)^-1: \n" << result;      
  
  // reorder components to match TrackingRecHit interface (CSCSegment isa TrackingRecHit)
  // i.e. slopes first, then positions 
  AlgebraicSymMatrix flipped = flipErrors( result );
    
  return flipped;
}


AlgebraicSymMatrix CSCSegFit::flipErrors( const SMatrixSym4& a ) { 
    
  // The CSCSegment needs the error matrix re-arranged to match
  // parameters in order (uz, vz, u0, v0) 
  // where uz, vz = slopes, u0, v0 = intercepts
    
  //  LogTrace("CSCSegFit") << "[CSCSegFit::flipErrors] input: \n" << a;      

  AlgebraicSymMatrix hold(4, 0. ); 
      
  for ( short j=0; j!=4; ++j) {
    for (short i=0; i!=4; ++i) {
      hold(i+1,j+1) = a(i,j); // SMatrix counts from 0, AlgebraicMatrix from 1
    }
  }

  //  LogTrace("CSCSegFit") << "[CSCSegFit::flipErrors] after copy:";
  //  LogTrace("CSCSegFit") << "(" << hold(1,1) << "  " << hold(1,2) << "  " << hold(1,3) << "  " << hold(1,4);
  //  LogTrace("CSCSegFit") << " " << hold(2,1) << "  " << hold(2,2) << "  " << hold(2,3) << "  " << hold(2,4);
  //  LogTrace("CSCSegFit") << " " << hold(3,1) << "  " << hold(3,2) << "  " << hold(3,3) << "  " << hold(3,4);
  //  LogTrace("CSCSegFit") << " " << hold(4,1) << "  " << hold(4,2) << "  " << hold(4,3) << "  " << hold(4,4) << ")";

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

  //  LogTrace("CSCSegFit") << "[CSCSegFit::flipErrors] after flip:";
  //  LogTrace("CSCSegFit") << "(" << hold(1,1) << "  " << hold(1,2) << "  " << hold(1,3) << "  " << hold(1,4);
  //  LogTrace("CSCSegFit") << " " << hold(2,1) << "  " << hold(2,2) << "  " << hold(2,3) << "  " << hold(2,4);
  //  LogTrace("CSCSegFit") << " " << hold(3,1) << "  " << hold(3,2) << "  " << hold(3,3) << "  " << hold(3,4);
  //  LogTrace("CSCSegFit") << " " << hold(4,1) << "  " << hold(4,2) << "  " << hold(4,3) << "  " << hold(4,4) << ")";

  return hold;
}
 
float CSCSegFit::xfit( float z ) const {
  //@@ ADD THIS TO EACH ACCESSOR OF FIT RESULTS?
  //  if ( !fitdone() ) fit();
  return intercept_.x() + uslope_ * z;
}

float CSCSegFit::yfit( float z ) const {
  return intercept_.y() + vslope_ * z;
}

float CSCSegFit::xdev( float x, float z ) const {
  return intercept_.x() + uslope_ * z - x;
}

float CSCSegFit::ydev( float y, float z ) const {
  return intercept_.y() + vslope_ * z - y;
}

float CSCSegFit::Rdev(float x, float y, float z) const {
  return sqrt ( xdev(x,z)*xdev(x,z) + ydev(y,z)*ydev(y,z) );
}

