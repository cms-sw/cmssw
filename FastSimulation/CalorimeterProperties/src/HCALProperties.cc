#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"
#include <cmath>
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"


HCALProperties::HCALProperties(const edm::ParameterSet& fastDet) : CalorimeterProperties()
{

    edm::ParameterSet fastDetHCAL= fastDet.getParameter<edm::ParameterSet>("HadronicCalorimeterProperties");
  hOPi = fastDetHCAL.getParameter<double>("HCAL_PiOverE");
  spotFrac= fastDetHCAL.getParameter<double>("HCAL_Sampling");
  HCALAeff_= fastDetHCAL.getParameter<double>("HCALAeff");
  HCALZeff_= fastDetHCAL.getParameter<double>("HCALZeff");
  HCALrho_= fastDetHCAL.getParameter<double>("HCALrho");
  HCALradiationLengthIncm_= fastDetHCAL.getParameter< double>("HCALradiationLengthIncm");
  HCALradLenIngcm2_= fastDetHCAL.getParameter<double>("HCALradLenIngcm2");
  HCALmoliereRadius_= fastDetHCAL.getParameter<double>("HCALmoliereRadius");
  HCALcriticalEnergy_= fastDetHCAL.getParameter<double>("HCALcriticalEnergy");
  HCALinteractionLength_= fastDetHCAL.getParameter<double>("HCALinteractionLength");
  etatow_=fastDetHCAL.getParameter<std::vector<double>>("HCALetatow");
  hcalDepthLam_=fastDetHCAL.getParameter<std::vector<double>>("HCALDepthLam");

  // in principle this splitting into 42 bins may change with future detectors, but let's add a protection to make sure that differences are not typos in the configuration file:
  if (etatow_.size() != 42) std::cout << " HCALProperties::eta2ieta - WARNING: here we expect 42 entries instead of " << etatow_.size() << "; is the change intentional?" << std::endl;
  // splitting of  28-th tower is taken into account (2.65-2.853-3.0)
  if (hcalDepthLam_.size() != etatow_.size()-1) std::cout << " HCALProperties::eta2ieta - WARNING: the sizes of HCALetatow and HCALDepthLam should differ by 1 unit! HCALDepthLam has size " << hcalDepthLam_.size()<< " and HCALetatow has size " << etatow_.size() << std::endl;
  

}

double HCALProperties::getHcalDepth(double eta) const{

  int  ieta = eta2ieta(eta); 

  /* 
  std::cout << " HCALProperties::getHcalDepth for eta = " << eta 
	    << "  returns lam.thickness = " << hcalDepthLam_[ieta] << std::endl;
  */

  return  hcalDepthLam_[ieta];

}



int HCALProperties::eta2ieta(double eta) const {
  // binary search in the array of towers eta edges
  int size = etatow_.size();

  double x = fabs(eta);
  int curr = size / 2;
  int step = size / 4;
  int iter;
  int prevdir = 0; 
  int actudir = 0; 

  for (iter = 0; iter < size ; iter++) {

    if( curr >= size || curr < 1 )
      std::cout <<  " HCALProperties::eta2ieta - wrong current index = "
		<< curr << " !!!" << std::endl;

    if ((x <= etatow_[curr]) && (x > etatow_[curr-1])) break;
    prevdir = actudir;
    if(x > etatow_[curr]) {actudir =  1;}
    else                 {actudir = -1;}
    if(prevdir * actudir < 0) { if(step > 1) step /= 2;}
    curr += actudir * step;
    if(curr > size) curr = size;
    else { if(curr < 1) {curr = 1;}}

    /*
    std::cout << " HCALProperties::eta2ieta  end of iter." << iter 
	      << " curr, etatow_[curr-1], etatow_[curr] = "
	      << curr << " " << etatow_[curr-1] << " " << etatow_[curr] << std::endl;
    */
    
  }

  /*
  std::cout << " HCALProperties::eta2ieta  for input x = " << x 
	    << "  found index = " << curr-1
	    << std::endl;
  */
  
  return curr-1;
}
