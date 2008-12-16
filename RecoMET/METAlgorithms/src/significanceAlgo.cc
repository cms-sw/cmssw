#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"
#include <iostream>
#include <string>
#include "TMath.h"
#include "math.h"
#include "TROOT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SignAlgoResolutions
// 
/**\class METSignificance SignAlgoResolutions.cc RecoMET/METAlgorithms/src/SignAlgoResolutions.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id: significanceAlgo.cc,v 1.6 2008/11/13 01:30:44 rcr Exp $
//
//

// //*** rotate a 2D matrix by angle theta **********************//
void
metsig::rotateMatrix( Double_t theta, TMatrixD &v)
{

  TMatrixD r(2,2);
  TMatrixD rInv(2,2);

  r(0,0) = cos(theta); r(0,1) = sin(theta); r(1,0) = -sin(theta); r(1,1) = cos(theta);
  rInv = r;
  rInv.Invert();
  //-- Rotate v --//
  v = rInv * v * r;
}
//************************************************************//

//************************************************************//
double 
metsig::ASignificance(const std::vector<SigInputObj>& EventVec, double &met_r, double &met_phi, double &met_set) 
{
  
  if(EventVec.size()<1) {
    //edm::LogWarning("SignCaloSpecificAlgo") << "Event Vector is empty!  Return significance -1";
    return(-1);
  }
  //=== Analytical Generation of Chisq Contours ===//
  double set_worker = 0;
  double xmet=0, ymet=0;
  double chisq0;

  //--- Temporary variables ---//
  TMatrixD v_tmp(2,2);
  TMatrixD v_tot(2,2);
  TVectorD metvec(2);

  //--- Initialize sum of rotated covariance matrices ---//
  v_tot(0,0)=0; v_tot(0,1)=0; v_tot(1,0)=0; v_tot(1,1)=0;

  //--- Loop over physics objects in the event ---//
  //  for(unsigned int objnum=1; objnum < EventVec.size(); objnum++ ) {
  for(std::vector<SigInputObj>::const_iterator obj = EventVec.begin(); obj!= EventVec.end(); ++obj){
    double et_tmp     = obj->get_energy();
    double phi_tmp    = obj->get_phi();
    double sigma_et   = obj->get_sigma_e();
    double sigma_tan  = obj->get_sigma_tan();

    double xval = et_tmp * cos( phi_tmp );
    double yval = et_tmp * sin( phi_tmp );
    xmet -= xval;
    ymet -= yval;
    set_worker += et_tmp;

    //-- Initialize covariance matrix --//
    v_tmp(0,0) = pow( sigma_et, 2);
    v_tmp(0,1) = 0;  v_tmp(1,0) = 0;
    v_tmp(1,1) = pow( sigma_tan, 2);
    
    //-- rotate matrix --//
    rotateMatrix(phi_tmp,v_tmp);

    //-- add to sum of rotated covariance matrices --//
    v_tot += v_tmp;
  }


  //--- Calculate magnitude and angle of MET, store in returned variables ---//
  met_r = sqrt(xmet*xmet + ymet*ymet);
  met_set = set_worker;
  //--- Ensure met_phi is in [-pi, pi] ---//

//   double tmp_met_phi;
//   if( (xmet) >=0.0 ) {
//     tmp_met_phi = TMath::ATan( (ymet) / (xmet) );
//   }
//   else {
//     if( (ymet) >=0.0 ) {
//       tmp_met_phi = TMath::ATan(ymet/xmet) + TMath::Pi();
//     }
//     else{ // => ymet<0
//       tmp_met_phi = TMath::ATan(ymet/xmet) - TMath::Pi();
//     }
//   }
//   met_phi = tmp_met_phi;

  met_phi= TMath::ATan2(ymet, xmet);
  
  //--- Calculate Significance ---//
  v_tot.Invert();
  metvec(0) = xmet; metvec(1) = ymet;
  chisq0 = metvec * (v_tot * metvec);
  double lnSignificance = chisq0;  

  return lnSignificance;
}
//*** End of ASignificance ********************************//
