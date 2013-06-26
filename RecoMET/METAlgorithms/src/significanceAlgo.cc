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
// $Id: significanceAlgo.cc,v 1.15 2010/04/20 14:56:07 fblekman Exp $
//
//

metsig::significanceAlgo::significanceAlgo():
  //  eventVec_(0),
  signifmatrix_(2,2),
  set_worker_(0),
  xmet_(0),
  ymet_(0)

{
  //  std::cout << "in constructor ! " << std::endl;
  signifmatrix_(0,0)=signifmatrix_(1,0)=signifmatrix_(0,1)=signifmatrix_(1,1)=0;
}


//******* Add an existing significance matrix to the algo, so that the vector sum can be continued. Only makes sense if matrix is empty or you want to purposefully increase uncertainties (for example in systematic studies)!
const void
metsig::significanceAlgo::addSignifMatrix(const TMatrixD &input){
  // check that the matrix is size 2:
  if(input.GetNrows()==2 && input.GetNcols()==2) {
    signifmatrix_+=input;
  }
  return;
}
////////////////////////
/// reset the signficance matrix (this is the most likely case), so that the vector sum can be continued

const void
metsig::significanceAlgo::setSignifMatrix(const TMatrixD &input,const double &met_r, const double &met_phi, const double &met_set){
  // check that the matrix is size 2:
  if(input.GetNrows()==2 && input.GetNcols()==2) {
    signifmatrix_=input;
    set_worker_=met_set;
    xmet_=met_r*cos(met_phi);
    ymet_=met_r*sin(met_phi);
  }
  return;
}

// ********destructor ********

metsig::significanceAlgo::~significanceAlgo(){
}

// //*** rotate a 2D matrix by angle theta **********************//

void
metsig::significanceAlgo::rotateMatrix( Double_t theta, TMatrixD &v)
{
  // I suggest not using this to rotate trivial matrices.
  TMatrixD r(2,2);
  TMatrixD rInv(2,2);

  r(0,0) = cos(theta); r(0,1) = sin(theta); r(1,0) = -sin(theta); r(1,1) = cos(theta);
  rInv = r;
  rInv.Invert();
  //-- Rotate v --//
  v = rInv * v * r;
}
//************************************************************//


const void 
metsig::significanceAlgo::subtractObjects(const std::vector<metsig::SigInputObj>& eventVec)
{ 
  TMatrixD v_tot = signifmatrix_;
  //--- Loop over physics objects in the event ---//
  //  for(unsigned int objnum=1; objnum < EventVec.size(); objnum++ ) {
  for(std::vector<SigInputObj>::const_iterator obj = eventVec.begin(); obj!= eventVec.end(); ++obj){
    double et_tmp     = obj->get_energy();
    double phi_tmp    = obj->get_phi();
    double sigma_et   = obj->get_sigma_e();
    double sigma_tan  = obj->get_sigma_tan();

    double cosphi=cos(phi_tmp);
    double sinphi=sin(phi_tmp);
    double xval = et_tmp * cosphi;
    double yval = et_tmp * sinphi;
    xmet_ += xval;
    ymet_ += yval;
    set_worker_ -= et_tmp;

    double sigma0_2=sigma_et*sigma_et;
    double sigma1_2=sigma_tan*sigma_tan;

    v_tot(0,0)-= sigma0_2*cosphi*cosphi + sigma1_2*sinphi*sinphi;
    v_tot(0,1)-= cosphi*sinphi*(sigma0_2 - sigma1_2);
    v_tot(1,0)-= cosphi*sinphi*(sigma0_2 - sigma1_2);
    v_tot(1,1)-= sigma1_2*cosphi*cosphi + sigma0_2*sinphi*sinphi;
    
  }
  signifmatrix_=v_tot;
}
//************************************************************//


const void 
metsig::significanceAlgo::addObjects(const std::vector<metsig::SigInputObj>& eventVec)
{ 
  TMatrixD v_tot = signifmatrix_;
  //--- Loop over physics objects in the event ---//
  //  for(unsigned int objnum=1; objnum < EventVec.size(); objnum++ ) {
  for(std::vector<SigInputObj>::const_iterator obj = eventVec.begin(); obj!= eventVec.end(); ++obj){
    double et_tmp     = obj->get_energy();
    double phi_tmp    = obj->get_phi();
    double sigma_et   = obj->get_sigma_e();
    double sigma_tan  = obj->get_sigma_tan();

    double cosphi=cos(phi_tmp);
    double sinphi=sin(phi_tmp);
    double xval = et_tmp * cosphi;
    double yval = et_tmp * sinphi;
    xmet_ -= xval;
    ymet_ -= yval;
    set_worker_ += et_tmp;

    double sigma0_2=sigma_et*sigma_et;
    double sigma1_2=sigma_tan*sigma_tan;

    v_tot(0,0)+= sigma0_2*cosphi*cosphi + sigma1_2*sinphi*sinphi;
    v_tot(0,1)+= cosphi*sinphi*(sigma0_2 - sigma1_2);
    v_tot(1,0)+= cosphi*sinphi*(sigma0_2 - sigma1_2);
    v_tot(1,1)+= sigma1_2*cosphi*cosphi + sigma0_2*sinphi*sinphi;
    
  }
  signifmatrix_=v_tot;
}

//************************************************************//
const double 
metsig::significanceAlgo::significance(double &met_r, double &met_phi, double &met_set) 
{
  
  if(signifmatrix_(0,0)==0 && signifmatrix_(1,1)==0 && signifmatrix_(1,0)==0 && signifmatrix_(0,1)==0){
    //edm::LogWarning("SignCaloSpecificAlgo") << "Event Vector is empty!  Return significance -1";
    return(-1);
  } 

  //--- Temporary variables ---//
 
  TMatrixD v_tot(2,2);
  TVectorD metvec(2);

 //--- Initialize sum of rotated covariance matrices ---//  
  v_tot=signifmatrix_;
  //  std::cout << "INPUT:\n"<< v_tot(0,0) << "," << v_tot(0,1) << "\n" << v_tot(1,0) << "," << v_tot(1,1) << std::endl;



  //--- Calculate magnitude and angle of MET, store in returned variables ---//
  met_r = sqrt(xmet_*xmet_ + ymet_*ymet_);
  met_set = set_worker_;
  met_phi= TMath::ATan2(ymet_, xmet_);

  // one other option: if particles cancel there could be small numbers.
  // this check fixes this, added by F.Blekman
  if(fabs(v_tot.Determinant())<0.000001)
    return -1;


  // save matrix into object:
  //  std::cout << "SAVED:\n"<< v_tot(0,0) << "," << v_tot(0,1) << "\n" << v_tot(1,0) << "," << v_tot(1,1) << std::endl;
  v_tot.Invert();
  //  std::cout << "INVERTED:\n"<< v_tot(0,0) << "," << v_tot(0,1) << "\n" << v_tot(1,0) << "," << v_tot(1,1) << std::endl;
  


  metvec(0) = xmet_; metvec(1) = ymet_;
  double lnSignificance = metvec * (v_tot * metvec);

  //  v_tot.Invert();
  //  std::cout << "INVERTED AGAIN:\n"<< v_tot(0,0) << "," << v_tot(0,1) << "\n" << v_tot(1,0) << "," << v_tot(1,1) << std::endl;
  return lnSignificance;
}
//*** End of Significance calculation ********************************//
