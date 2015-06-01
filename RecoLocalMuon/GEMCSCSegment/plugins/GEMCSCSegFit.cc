// ------------------------- //
// --> GEMCSCSegFit.cc 
// Created:  21.04.2015
// --> Based on CSCSegFit.cc 
// with last mod: 03.02.2015
// as found in 750pre2 rel
// ------------------------- //

#include "RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegFit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void GEMCSCSegFit::fit(void) {
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fit] - start the fitting fun :: nhits = "<<nhits();
  if ( fitdone() ) return; // don't redo fit unnecessarily
  short n = nhits();
  switch ( n ) {
  case 1:
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fit] - cannot fit just 1 hit!!";
    break;
  case 2:
    fit2();
    break;
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
    fitlsq();
    break;
  default:
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fit] - cannot fit more than 8 hits!!";
  }  
}

void GEMCSCSegFit::fit2(void) {

  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fit2] - start fit2()";
  // Just join the two points
  // Equation of straight line between (x1, y1) and (x2, y2) in xy-plane is
  //       y = mx + c
  // with m = (y2-y1)/(x2-x1)
  // and  c = (y1*x2-x2*y1)/(x2-x1)


  // 1) Check whether hits are on the same layer
  // -------------------------------------------
  std::vector<const TrackingRecHit*>::const_iterator ih = hits_.begin();
  // layer numbering: GEM: (1,2) CSC (3,4,5,6,7,8)
  int il1 = 0, il2 = 0;
  DetId d1 = DetId((*ih)->rawId());
  // check whether first hit is GEM or CSC
  if (d1.subdetId() == MuonSubdetId::GEM) {
    il1 = GEMDetId(d1).layer();
  }
  else if (d1.subdetId() == MuonSubdetId::CSC) {
    il1 = CSCDetId(d1).layer() + 2;
  }
  else { edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - TrackingRecHit is not in GEM or CSC subdetector"; }
  const TrackingRecHit& h1 = (**ih);
  ++ih;
  DetId d2 = DetId((*ih)->rawId());
  // check whether second hit is GEM or CSC
  if (d2.subdetId() == MuonSubdetId::GEM) {
    il2 = GEMDetId(d2).layer();
  }
  else if (d2.subdetId() == MuonSubdetId::CSC) {
    il2 = GEMDetId(d2).layer() + 2;
  }
  else { edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - TrackingRecHit is not in GEM or CSC subdetector"; }
  const TrackingRecHit& h2 = (**ih);
  // Skip if on same layer, but should be impossible :)
  if (il1 == il2) {
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - 2 hits on same layer!!";
    return;
  }


  // 2) Global Positions of hit 1 and 2 and
  //    Local  Positions of hit 1 and 2 w.r.t. reference CSC Chamber Frame 
  // ---------------------------------------------------------------------
  GlobalPoint h1glopos, h2glopos;
  // global position hit 1
  if(d1.subdetId() == MuonSubdetId::GEM) {
    const GEMEtaPartition* roll1 = gemetapartition(GEMDetId(d1));
    h1glopos = roll1->toGlobal(h1.localPosition());
  }
  else if(d1.subdetId() == MuonSubdetId::CSC) {
    const CSCLayer* layer1 = cscchamber(CSCDetId(d1))->layer(CSCDetId(d1).layer());
    h1glopos = layer1->toGlobal(h1.localPosition());
  }
  else { edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - TrackingRecHit is not in GEM or CSC subdetector"; }
  // global position hit 2
  if(d2.subdetId() == MuonSubdetId::GEM) {
    const GEMEtaPartition* roll2 = gemetapartition(GEMDetId(d2));
    h2glopos = roll2->toGlobal(h2.localPosition());
  }
  else if(d2.subdetId() == MuonSubdetId::CSC) {
    const CSCLayer* layer2 = cscchamber(CSCDetId(d2))->layer(CSCDetId(d2).layer());
    h2glopos = layer2->toGlobal(h2.localPosition());
  }
  else { edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - TrackingRecHit is not in GEM or CSC subdetector"; }
  // local positions hit 1 and 2 w.r.t. ref CSC Chamber Frame
  // We want hit wrt chamber (and local z will be != 0)
  LocalPoint h1pos = refcscchamber()->toLocal(h1glopos);  
  LocalPoint h2pos = refcscchamber()->toLocal(h2glopos);  


  // 3) Now make straight line between the two points in local coords
  // ----------------------------------------------------------------    
  float dz = h2pos.z()-h1pos.z();
  if(dz != 0) {
    uslope_ = ( h2pos.x() - h1pos.x() ) / dz ;
    vslope_ = ( h2pos.y() - h1pos.y() ) / dz ;
  }
  float uintercept = ( h1pos.x()*h2pos.z() - h2pos.x()*h1pos.z() ) / dz;
  float vintercept = ( h1pos.y()*h2pos.z() - h2pos.y()*h1pos.z() ) / dz;
  intercept_ = LocalPoint( uintercept, vintercept, 0.);

  setOutFromIP();

  //@@ NOT SURE WHAT IS SENSIBLE FOR THESE...
  chi2_ = 0.;
  ndof_ = 0;

  fitdone_ = true;
}


void GEMCSCSegFit::fitlsq(void) {
  
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fitlsq] - start fitlsq";
  // Linear least-squares fit to up to 6 CSC rechits, one per layer in a CSC
  // and up to 2 GEM rechits in a GEM superchamber
  // (we can later later on go up to 4 hits in case of an overlapping chamber)
  // (... and if there is an overlap in the GEM chambers, than maybe we also
  //  have an overlap in the CSC chambers... maybe we can also benefit there)

  // Comments below taken from CSCSegFit algorithm.
  // Comments adapted from original  CSCSegAlgoSK algorithm.
  
  // Fit to the local x, y rechit coordinates in z projection
  // The CSC & GEM strip measurement controls the precision of x
  // The CSC wire measurement & GEM eta-partition controls the precision of y. 
  // Typical precision CSC: u (strip, sigma~200um), v (wire, sigma~1cm)
  // Typical precision GEM: u (strip, sigma~250um), v (eta-part, sigma~10-20cm)
  
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

  std::vector<const TrackingRecHit*>::const_iterator ih = hits_.begin();

  // LogDebug :: Loop over the TrackingRecHits and print the GEM and CSC Hits  
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) 
    {
      edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fitlsq] - looping over TrackingRecHits";
      const TrackingRecHit& hit = (**ih);
      DetId d = DetId(hit.rawId());
      edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fitlsq] - Tracking RecHit in detid ("<<d.rawId()<<")";
      if(d.subdetId() == MuonSubdetId::GEM) {
	edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fitlsq] - Tracking RecHit is a GEM Hit in detid ("<<d.rawId()<<")";
	edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fitlsq] - GEM DetId ("<<GEMDetId(d.rawId())<<")";
      }
      else if(d.subdetId() == MuonSubdetId::CSC) {
	edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fitlsq] - Tracking RecHit is a CSC Hit in detid ("<<d.rawId()<<")";
	edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fitlsq] - CSC DetId ("<<CSCDetId(d.rawId())<<")";
      }
      else { edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - TrackingRecHit is not in GEM or CSC subdetector"; }
    }

  // Loop over the TrackingRecHits and make small (2x2) matrices used to fill the blockdiagonal covariance matrix E^-1
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) 
    {
      const TrackingRecHit& hit = (**ih);
      GlobalPoint gp;
      DetId d = DetId(hit.rawId());
      if(d.subdetId() == MuonSubdetId::GEM) 
	{
	  const GEMEtaPartition* roll = gemetapartition(GEMDetId(d));
	  gp = roll->toGlobal(hit.localPosition());
	}
      else if(d.subdetId() == MuonSubdetId::CSC) 
	{
	  const CSCLayer* layer = cscchamber(CSCDetId(d))->layer(CSCDetId(d).layer());
	  gp = layer->toGlobal(hit.localPosition());
	}
      else { edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - TrackingRecHit is not in GEM or CSC subdetector"; }
      LocalPoint lp = refcscchamber()->toLocal(gp); 

      // LogDebug
      std::stringstream lpss; lpss<<lp; std::string lps = lpss.str();
      std::stringstream gpss; gpss<<gp; std::string gps = gpss.str();
      edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fitlsq] - Tracking RecHit global position "<<std::setw(30)<<gps<<" and local position "<<std::setw(30)<<lps
				       <<" wrt reference csc chamber "<<refcscchamber()->id().rawId()<<" = "<<refcscchamber()->id();
      
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

      edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fit] 2x2 covariance matrix for this Tracking RecHit :: [[" << IC(0,0) <<", "<< IC(0,1) <<"]["<< IC(1,0) <<","<<IC(1,1)<<"]]";
      
      // Invert covariance matrix (and trap if it fails!)
      bool ok = IC.Invert();
      if ( !ok ) 
	{
	  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fit] Failed to invert covariance matrix: \n" << IC;      
	  return; // MATRIX INVERSION FAILED ... QUIT VOID FUNCTION
	}

      // M = (AT E A)
      // B = (AT E m)
      // for now fill only with sum of blockdiagonal 
      // elements of (E^-1)_i = IC_i for hit i
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
      
    } // End Loop over the TrackingRecHits to make the block matrices to be filled in M and B
  
  SVector4 p;
  bool ok = M.Invert();
  if (!ok )
    {
      edm::LogVerbatim("GEMCSCSegment|GEMCSCSegFit") << "[GEMCSCSegFit::fit] Failed to invert matrix: \n" << M;
      return; // MATRIX INVERSION FAILED ... QUIT VOID FUNCTION 
    }
  else 
    {
      p = M * B;
    }

  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::fit] p = " 
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

}



void GEMCSCSegFit::setChi2(void) {
  
  double chsq = 0.;
  bool gem = false;

  std::vector<const TrackingRecHit*>::const_iterator ih;
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {

    const TrackingRecHit& hit = (**ih);
    GlobalPoint gp;
    DetId d = DetId(hit.rawId());
    if(d.subdetId() == MuonSubdetId::GEM) {
      const GEMEtaPartition* roll = gemetapartition(GEMDetId(d));
      gp = roll->toGlobal(hit.localPosition());
      gem = true;
    }
    else if(d.subdetId() == MuonSubdetId::CSC) {
      const CSCLayer* layer = cscchamber(CSCDetId(d))->layer(CSCDetId(d).layer());
      gp = layer->toGlobal(hit.localPosition());
      gem = false;
    }
    else { edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - TrackingRecHit is not in GEM or CSC subdetector"; }
    LocalPoint lp = refcscchamber()->toLocal(gp);

    // LogDebug
    std::stringstream lpss; lpss<<lp; std::string lps = lpss.str();
    std::stringstream gpss; gpss<<gp; std::string gps = gpss.str();
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] - Tracking RecHit in "<<(gem? "GEM":"CSC")<<" global position "<<std::setw(30)<<gps<<" and local position "<<std::setw(30)<<lps
				       <<" wrt reference csc chamber "<<refcscchamber()->id().rawId()<<" = "<<refcscchamber()->id();
    
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    double du = intercept_.x() + uslope_ * z - u;
    double dv = intercept_.y() + vslope_ * z - v;
    
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] u, v, z = " << u << ", " << v << ", " << z;

    SMatrixSym2 IC; // 2x2, init to 0

    IC(0,0) = hit.localPositionError().xx();
    //    IC(0,1) = hit.localPositionError().xy();
    IC(1,0) = hit.localPositionError().xy();
    IC(1,1) = hit.localPositionError().yy();
    //    IC(1,0) = IC(0,1);

    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] IC before = \n" << IC;

    // Invert covariance matrix
    bool ok = IC.Invert();
    if (!ok ){
      edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] Failed to invert covariance matrix: \n" << IC;
      return; // MATRIX INVERSION FAILED ... QUIT VOID FUNCTION
    }
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] IC after = \n" << IC;
    chsq += du*du*IC(0,0) + 2.*du*dv*IC(0,1) + dv*dv*IC(1,1);
    // LogDebug
    edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] Contribution of this Tracking RecHit to Chi2: du^2*D(1,1) + 2*du*dv*D(1,2) + dv^2*D(2,2) = " 
				       << du*du <<"*"<<IC(0,0)<<" + 2.*"<<du<<"*"<<dv<<"*"<<IC(0,1)<<" + "<<dv*dv<<"*"<<IC(1,1)<<" = "<<du*du*IC(0,0) + 2.*du*dv*IC(0,1) + dv*dv*IC(1,1);
  }
  // LogDebug
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] Total Chi2 = "<<chsq;
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] Total NDof = "<<2.*hits_.size() - 4;
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] Total Chi2/NDof = "<<((hits_.size()>2)?(chsq/(2.*hits_.size() - 4)):(0.0));

  // fill member variables
  chi2_ = chsq;
  ndof_ = 2.*hits_.size() - 4;

  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::setChi2] chi2 = " << chi2_ << "/" << ndof_ << " dof";

}




GEMCSCSegFit::SMatrixSym16 GEMCSCSegFit::weightMatrix() {
  
  bool ok = true;

  SMatrixSym16 matrix = ROOT::Math::SMatrixIdentity(); 
  // for CSC segment ::     max 6 rechits => 12x12, 
  // for GEM-CSC segment :: max 8 rechits => 16x16
  // for all :: init to 1's on diag

  int row = 0;
  
  for (std::vector<const TrackingRecHit*>::const_iterator it = hits_.begin(); it != hits_.end(); ++it) 
    {
   
      const TrackingRecHit& hit = (**it);

      // Note scaleXError allows rescaling the x error if necessary

      matrix(row, row)   = scaleXError()*hit.localPositionError().xx();
      matrix(row, row+1) = hit.localPositionError().xy();
      ++row;
      matrix(row, row-1) = hit.localPositionError().xy();
      matrix(row, row)   = hit.localPositionError().yy();
      ++row;
    }

  ok = matrix.Invert(); // invert in place
  if ( !ok ) 
    {
      edm::LogVerbatim("GEMCSCSegment|GEMCSCSegFit") << "[GEMCSCSegFit::weightMatrix] Failed to invert matrix: \n" << matrix;      
      SMatrixSym16 emptymatrix = ROOT::Math::SMatrixIdentity();
      return emptymatrix; // return (empty) identity matrix if matrix inversion failed 
    }
  return matrix;
}




GEMCSCSegFit::SMatrix16by4 GEMCSCSegFit::derivativeMatrix() {
  
  SMatrix16by4 matrix; // 16x4, init to 0
  int row = 0;
  
  for(std::vector<const TrackingRecHit*>::const_iterator it = hits_.begin(); it != hits_.end(); ++it) {
    
    const TrackingRecHit& hit = (**it);
    GlobalPoint gp;
    DetId d = DetId(hit.rawId());
    if(d.subdetId() == MuonSubdetId::GEM) {
      const GEMEtaPartition* roll = gemetapartition(GEMDetId(d));
      gp = roll->toGlobal(hit.localPosition());
    }
    else if(d.subdetId() == MuonSubdetId::CSC) {
      const CSCLayer* layer = cscchamber(CSCDetId(d))->layer(CSCDetId(d).layer());
      gp = layer->toGlobal(hit.localPosition());
    }
    else { edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit:fit2] - TrackingRecHit is not in GEM or CSC subdetector"; }
    LocalPoint lp = refcscchamber()->toLocal(gp);
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


void GEMCSCSegFit::setOutFromIP() {
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
  
  double globalZpos    = ( refcscchamber()->toGlobal( intercept_ ) ).z();
  double globalZdir    = ( refcscchamber()->toGlobal( localDir  ) ).z();
  double directionSign = globalZpos * globalZdir;
  localdir_ = (directionSign * localDir ).unit();
}



AlgebraicSymMatrix GEMCSCSegFit::covarianceMatrix() {
  
  SMatrixSym16 weights = weightMatrix();
  SMatrix16by4 A = derivativeMatrix();
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::covarianceMatrix] weights matrix W: \n" << weights;      
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::covarianceMatrix] derivatives matrix A: \n" << A;      

  // (AT W A)^-1
  // e.g. See http://www.phys.ufl.edu/~avery/fitting.html, part I

  bool ok;
  SMatrixSym4 result =  ROOT::Math::SimilarityT(A, weights);
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::covarianceMatrix] (AT W A): \n" << result;      
  ok = result.Invert(); // inverts in place
  if ( !ok ) {
    edm::LogVerbatim("GEMCSCSegment|GEMCSCSegFit") << "[GEMCSCSegFit::calculateError] Failed to invert matrix: \n" << result;
    AlgebraicSymMatrix emptymatrix(4, 0. );
    return emptymatrix; // return empty matrix if matrix inversion failed
  }
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::covarianceMatrix] (AT W A)^-1: \n" << result;      
  
  // reorder components to match TrackingRecHit interface (GEMCSCSegment isa TrackingRecHit)
  // i.e. slopes first, then positions 
  AlgebraicSymMatrix flipped = flipErrors( result );
    
  return flipped;
}


AlgebraicSymMatrix GEMCSCSegFit::flipErrors( const SMatrixSym4& a ) { 
    
  // The GEMCSCSegment needs the error matrix re-arranged to match
  // parameters in order (uz, vz, u0, v0) 
  // where uz, vz = slopes, u0, v0 = intercepts
    
  edm::LogVerbatim("GEMCSCSegFit") << "[GEMCSCSegFit::flipErrors] input: \n" << a;      

  AlgebraicSymMatrix hold(4, 0. ); 
      
  for ( short j=0; j!=4; ++j) {
    for (short i=0; i!=4; ++i) {
      hold(i+1,j+1) = a(i,j); // SMatrix counts from 0, AlgebraicMatrix from 1
    }
  }

  //  LogTrace("GEMCSCSegFit") << "[GEMCSCSegFit::flipErrors] after copy:";
  //  LogTrace("GEMCSCSegFit") << "(" << hold(1,1) << "  " << hold(1,2) << "  " << hold(1,3) << "  " << hold(1,4);
  //  LogTrace("GEMCSCSegFit") << " " << hold(2,1) << "  " << hold(2,2) << "  " << hold(2,3) << "  " << hold(2,4);
  //  LogTrace("GEMCSCSegFit") << " " << hold(3,1) << "  " << hold(3,2) << "  " << hold(3,3) << "  " << hold(3,4);
  //  LogTrace("GEMCSCSegFit") << " " << hold(4,1) << "  " << hold(4,2) << "  " << hold(4,3) << "  " << hold(4,4) << ")";

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

  //  LogTrace("GEMCSCSegFit") << "[GEMCSCSegFit::flipErrors] after flip:";
  //  LogTrace("GEMCSCSegFit") << "(" << hold(1,1) << "  " << hold(1,2) << "  " << hold(1,3) << "  " << hold(1,4);
  //  LogTrace("GEMCSCSegFit") << " " << hold(2,1) << "  " << hold(2,2) << "  " << hold(2,3) << "  " << hold(2,4);
  //  LogTrace("GEMCSCSegFit") << " " << hold(3,1) << "  " << hold(3,2) << "  " << hold(3,3) << "  " << hold(3,4);
  //  LogTrace("GEMCSCSegFit") << " " << hold(4,1) << "  " << hold(4,2) << "  " << hold(4,3) << "  " << hold(4,4) << ")";

  return hold;
}
 
float GEMCSCSegFit::xfit( float z ) const {
  //@@ ADD THIS TO EACH ACCESSOR OF FIT RESULTS?
  //  if ( !fitdone() ) fit();
  return intercept_.x() + uslope_ * z;
}

float GEMCSCSegFit::yfit( float z ) const {
  return intercept_.y() + vslope_ * z;
}

float GEMCSCSegFit::xdev( float x, float z ) const {
  return intercept_.x() + uslope_ * z - x;
}

float GEMCSCSegFit::ydev( float y, float z ) const {
  return intercept_.y() + vslope_ * z - y;
}

float GEMCSCSegFit::Rdev(float x, float y, float z) const {
  return sqrt ( xdev(x,z)*xdev(x,z) + ydev(y,z)*ydev(y,z) );
}

