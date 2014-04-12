#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibrationHF.h"
#include <TMath.h>
#include <math.h>
#include <vector>
#include <TF1.h>

using namespace std;
using namespace edm;

PFEnergyCalibrationHF::PFEnergyCalibrationHF() {

  calibHF_use_ = false;
  calibHF_eta_step_.push_back(0.00);
  calibHF_eta_step_.push_back(2.90);
  calibHF_eta_step_.push_back(3.00);
  calibHF_eta_step_.push_back(3.20);
  calibHF_eta_step_.push_back(4.20);
  calibHF_eta_step_.push_back(4.40);
  calibHF_eta_step_.push_back(4.60);
  calibHF_eta_step_.push_back(4.80);
  calibHF_eta_step_.push_back(5.20);
  calibHF_eta_step_.push_back(5.40);
  for(unsigned int i=0;i< calibHF_eta_step_.size();++i){
    calibHF_a_EMonly_.push_back(1.00);
    calibHF_b_HADonly_.push_back(1.00);
    calibHF_a_EMHAD_.push_back(1.00);
    calibHF_b_EMHAD_.push_back(1.00);
  }
  
}

PFEnergyCalibrationHF::PFEnergyCalibrationHF(
					     bool calibHF_use, 
					     const std::vector<double>& calibHF_eta_step,
					     const std::vector<double>& calibHF_a_EMonly,
					     const std::vector<double>& calibHF_b_HADonly,
					     const std::vector<double>& calibHF_a_EMHAD,
					     const std::vector<double>& calibHF_b_EMHAD) {

  calibHF_use_ = calibHF_use    ;
  calibHF_eta_step_  = calibHF_eta_step;
  calibHF_a_EMonly_  = calibHF_a_EMonly;
  calibHF_b_HADonly_ = calibHF_b_HADonly;
  calibHF_a_EMHAD_   = calibHF_a_EMHAD;
  calibHF_b_EMHAD_   = calibHF_b_EMHAD;
}

 
 


PFEnergyCalibrationHF::~PFEnergyCalibrationHF()
{
//--- nothing to be done yet  
}


double PFEnergyCalibrationHF::energyEm(double uncalibratedEnergyECAL, 
			      double eta, double phi)  {

  double calibrated = 0.0; 
  //find eta bin.  default : 0.00;2.90;3.00;3.20;4.20;4.40;4.60;4.80;5.20;5.40;
  int ietabin = 0;
  for(unsigned int i=0;i< calibHF_eta_step_.size();++i){
    if((fabs(eta))>=calibHF_eta_step_[i]){
      ietabin = i;
    }
  }
  calibrated = uncalibratedEnergyECAL * calibHF_a_EMonly_[ietabin];
  return calibrated; 

 


  // return calibrated;
}


double PFEnergyCalibrationHF::energyHad(double uncalibratedEnergyHCAL, 
			      double eta, double phi)  {

  double calibrated = 0.0; 
  //find eta bin.  default : 0.00;2.90;3.00;3.20;4.20;4.40;4.60;4.80;5.20;5.40;
  int ietabin = 0;
  for(unsigned int i=0;i< calibHF_eta_step_.size();++i){
    if((fabs(eta))>=calibHF_eta_step_[i]){
      ietabin = i;
    }
  }
  calibrated = uncalibratedEnergyHCAL * calibHF_b_HADonly_[ietabin];
  return calibrated; 
}

double PFEnergyCalibrationHF::energyEmHad(double uncalibratedEnergyECAL, 
					  double uncalibratedEnergyHCAL, 
					  double eta, double phi){

  double calibrated = 0.0; 
  //find eta bin.  default : 0.00;2.90;3.00;3.20;4.20;4.40;4.60;4.80;5.20;5.40+;
  int ietabin = 0;
  for(unsigned int i=0;i< calibHF_eta_step_.size();++i){
    if((fabs(eta))>=calibHF_eta_step_[i]){
      ietabin = i;
    }
  }
  calibrated = uncalibratedEnergyECAL * calibHF_a_EMHAD_[ietabin] + uncalibratedEnergyHCAL * calibHF_b_EMHAD_[ietabin];
  return calibrated; 
}

std::ostream& operator<<(std::ostream& out, 
			 const PFEnergyCalibrationHF& calib) {


  if(!out ) return out;
  out<<"PFEnergyCalibrationHF -- "<<endl;
  int ii = 0;
  for(std::vector<double>::const_iterator iteta =(calib.getcalibHF_eta_step()).begin(); 
      iteta !=(calib.getcalibHF_eta_step()).end() ;++iteta){
    //double currenteta = *iteta;
    out<<" i "<<ii<<",";
    out<<"use "<<calib.getcalibHF_use()<<",";
    out<<"currenteta "<<calib.getcalibHF_eta_step()[ii]<<",";
    out<<"calibHF_a_EMonly_ "<<calib.getcalibHF_a_EMonly()[ii]<<",";
    out<<"calibHF_b_HADonly_ "<<calib.getcalibHF_b_HADonly()[ii]<<",";
    out<<"calibHF_a_EMHAD_ "<<calib.getcalibHF_a_EMHAD()[ii]<<",";
    out<<"calibHF_b_EMHAD_ "<<calib.getcalibHF_b_EMHAD()[ii]<<",";
    out<<endl;
    ii++;
  }
  
  

  return out;
}

