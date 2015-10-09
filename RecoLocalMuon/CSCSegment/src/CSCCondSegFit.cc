// CSCCondSegFit.cc 
// Last mod: 23.01.2015

#include "RecoLocalMuon/CSCSegment/src/CSCCondSegFit.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

void CSCCondSegFit::fit( bool condpass1, bool condpass2 ) {

  // See base class CSCSegFit::fit for detailed explanation of algorithm.
  // This is exactly the same, except it adds two conditioning passes
  // which can improve the robustness of the calculations.  
  
  lex_.clear(); // Vector of the error matrix (only xx)
  
  if(condpass1 && !condpass2){
    correctTheCovX();
    if(lex_.size()!=hits_.size()){
      LogTrace("CSCSegment|CSC") << "[CSCSegFit::fit] lex_.size()!=hits_.size() ALARM! Please inform CSC DPG " << std::endl;
    }
  }

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
    
    // Adjust covariance matrix elements to improve numerical stability?
    if(condpass1 && !condpass2){
      IC(0,0) = lex_.at(ih-hits_.begin());
    }
    else {
      IC(0,0) = hit.localPositionError().xx();
    }
    IC(1,1) = hit.localPositionError().yy();
    //@@ NOT SURE WHICH OFF-DIAGONAL ELEMENT MUST BE DEFINED BUT (1,0) WORKS
    //@@ (and SMatrix enforces symmetry)
    IC(1,0) = hit.localPositionError().xy();
    // IC(0,1) = IC(1,0);
    
    // Correct the covariance matrix to improve numerical stability?
    if(condpass2){
      correctTheCovMatrix(IC);
    }
    
    // Invert covariance matrix (and trap if it fails!)
    bool ok = IC.Invert();
    if ( !ok ) {
      LogTrace("CSCSegment|CSC") << "[CSCSegFit::fit] Failed to invert covariance matrix: \n" << IC;      
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
    LogTrace("CSCSegment|CSC") << "[CSCSegFit::fit] Failed to invert matrix: \n" << M;
    //    return ok; //@@ SHOULD PASS THIS BACK TO CALLER?
  }
  else {
    p = M * B;
  }

  //  LogTrace("CSCSegFit") << "[CSCSegFit::fit] p = " 
  //        << p(0) << ", " << p(1) << ", " << p(2) << ", " << p(3);
  
  // fill member variables  (note origin has local z = 0)
  intercept_ = LocalPoint(p(0), p(1), 0.);
   
  // localdir_
  uslope_ = p(2);
  vslope_ = p(3);
  setOutFromIP();
  
  // calculate chi2 of fit
  setChi2( condpass1, condpass2 );

}



void CSCCondSegFit::setChi2( bool condpass1, bool condpass2 ) {
  
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

    if(condpass1 && !condpass2){
      IC(0,0) = lex_.at(ih-hits_.begin());
    }
    else{
      IC(0,0) = hit.localPositionError().xx();
    }
    //    IC(0,1) = hit.localPositionError().xy();
    IC(1,0) = hit.localPositionError().xy();
    IC(1,1) = hit.localPositionError().yy();
    //    IC(1,0) = IC(0,1);

    //    LogTrace("CSCSegFit") << "[CSCSegFit::setChi2] IC before = \n" << IC;

    /// Correct the cov matrix
    if(condpass2){
      correctTheCovMatrix(IC);
    }
    
    // Invert covariance matrix
    bool ok = IC.Invert();
    if (!ok ){
      LogTrace("CSCSegment|CSC") << "[CSCSegFit::setChi2] Failed to invert covariance matrix: \n" << IC;
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



void CSCCondSegFit::correctTheCovX(void){
  std::vector<double> uu, vv, zz;  // Vectors of coordinates

  lex_.clear(); // x component of local error for each hit
  
  double sum_U_err=0.0;
  double sum_Z_U_err=0.0; 
  double sum_Z2_U_err=0.0;
  double sum_U_U_err=0.0;
  double sum_UZ_U_err=0.0;
  std::vector<double> chiUZind;
  std::vector<double>::iterator chiContribution;
  double chiUZ=0.0;
  CSCSetOfHits::const_iterator ih = hits_.begin();
  for (ih = hits_.begin(); ih != hits_.end(); ++ih) {
    const CSCRecHit2D& hit = (**ih);
    lex_.push_back(hit.localPositionError().xx());

    const CSCLayer* layer  = chamber()->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint  lp         = chamber()->toLocal(gp); 
    // Local position of hit w.r.t. chamber
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    uu.push_back(u); 
    vv.push_back(v); 
    zz.push_back(z);
    // Prepare the sums for the standard linear fit
    sum_U_err += 1./lex_.back();
    sum_Z_U_err += z/lex_.back();
    sum_Z2_U_err += (z*z)/lex_.back();
    sum_U_U_err += u/lex_.back();
    sum_UZ_U_err += (u*z)/lex_.back();
  }
 
  /// Make a one dimensional fit in U-Z plane (i.e. chamber local x-z)
  /// U=U0+UZ*Z fit parameters
  
  double denom=sum_U_err*sum_Z2_U_err-pow(sum_Z_U_err,2);
  double U0=(sum_Z2_U_err*sum_U_U_err-sum_Z_U_err*sum_UZ_U_err)/denom;
  double UZ=(sum_U_err*sum_UZ_U_err-sum_Z_U_err*sum_U_U_err)/denom;
  
  /// Calculate the fit line trace  
  /// Calculate one dimensional chi^2 and normalize the errors if needed
  
  for(unsigned i=0; i<uu.size(); ++i){
    double uMean = U0+UZ*zz[i];
    chiUZind.push_back((pow((uMean-uu[i]),2))/lex_[i]);
    chiUZ += (pow((uMean-uu[i]),2))/lex_[i];
  }
  chiUZ = chiUZ/(uu.size()-2);
  
  if(chiUZ>=chi2Norm_){
    double chi2uCorrection = chiUZ/chi2Norm_;
    for(unsigned i=0; i<uu.size(); ++i)
      lex_[i]=lex_[i]*chi2uCorrection;
      setScaleXError(chi2uCorrection);
  }
    
  // Find most deviant hit 
    chiContribution=max_element(chiUZind.begin(),chiUZind.end());
    worstHit_ = chiContribution - chiUZind.begin();
 
}

void CSCCondSegFit::correctTheCovMatrix(SMatrixSym2 &IC){
  //double condNumberCorr1=0.0;
  double condNumberCorr2=0.0; 
  double detCov=0.0;
  double diag1=0.0;
  double diag2=0.0;
  double IC_01_corr=0.0;
  double  IC_00_corr=0.0;
  if(!covToAnyNumberAll_){
    //condNumberCorr1=condSeed1_*IC(1,1);
    condNumberCorr2=condSeed2_*IC(1,1);
    diag1=IC(0,0)*IC(1,1);
    diag2=IC(0,1)*IC(0,1);
    detCov=fabs(diag1-diag2);
    if((diag1<condNumberCorr2)&&(diag2<condNumberCorr2)){
	  if(covToAnyNumber_)
	    IC(0,1)=covAnyNumber_;
	  else {	
	    IC_00_corr=condSeed1_+fabs(IC(0,1))/IC(1,1);
	    IC(0,0)=IC_00_corr;
	  }
    }
    
    if(((detCov<condNumberCorr2)&&(diag1>condNumberCorr2))||
       ((diag2>condNumberCorr2)&&(detCov<condNumberCorr2) )){
      if(covToAnyNumber_)
	    IC(0,1)=covAnyNumber_;
      else {	
	    IC_01_corr=sqrt(fabs(diag1-condNumberCorr2));
	    if(IC(0,1)<0)
	      IC(0,1)=(-1)*IC_01_corr;
	    else
	      IC(0,1)=IC_01_corr;
      }
    }
  }
  else{
    IC(0,1)=covAnyNumber_;
  }
  // With SMatrix formulation. the following might be unnecessary since it might
  // enforce the symmetry. But can't hurt to leave it (it WAS a real bug fix for
  // the original CLHEP formulation.)
  IC(1,0) = IC(0,1); //@@ ADDED TO MAINTAIN SYMMETRY OF IC 
}
