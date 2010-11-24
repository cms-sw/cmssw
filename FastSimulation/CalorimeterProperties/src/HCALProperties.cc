#include "FWCore/ParameterSet/interface/ParameterSet.h"

//This class header
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"
#include <cmath>
#include <iostream>


HCALProperties::HCALProperties(const edm::ParameterSet& fastDet) : CalorimeterProperties()
{
  hOPi = fastDet.getParameter<double>("HCAL_PiOverE");
  spotFrac= fastDet.getParameter<double>("HCAL_Sampling");

  // splitting of  28-th tower is taken into account (2.65-2.853-3.0)
  double etatow_ [42] = {
    0.000, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.870,    0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830,    1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853, 3.000, 3.139, 3.314, 3.489,    3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191
  };

  double hcalDepthLam_ [41] = { 
    8.930, 9.001, 9.132, 8.912, 8.104, 8.571, 8.852, 9.230, 9.732, 10.29,          10.95, 11.68, 12.49, 12.57, 12.63,  6.449, 5.806, 8.973, 8.934,  8.823,          8.727, 8.641, 8.565, 8.496, 8.436, 8.383, 8.346, 8.307, 8.298,  8.281,          9.442, 9.437, 9.432, 9.429, 9.432, 9.433, 9.430, 9.437, 9.442,  9.446, 9.435  };

  for (int i = 0; i < 42; i++) { etatow[i] = etatow_[i];}
  for (int i = 0; i < 41; i++) { hcalDepthLam[i] = hcalDepthLam_[i];}

}

double HCALProperties::getHcalDepth(double eta) const{

  int  ieta = eta2ieta(eta); 

  /* 
  std::cout << " HCALProperties::getHcalDepth for eta = " << eta 
	    << "  returns lam.thickness = " << hcalDepthLam[ieta] << std::endl;
  */

  return  hcalDepthLam[ieta];

}



int HCALProperties::eta2ieta(double eta) const {
  // binary search in the array of towers eta edges
  int size = 42;

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

    if ((x <= etatow[curr]) && (x > etatow[curr-1])) break;
    prevdir = actudir;
    if(x > etatow[curr]) {actudir =  1;}
    else                 {actudir = -1;}
    if(prevdir * actudir < 0) { if(step > 1) step /= 2;}
    curr += actudir * step;
    if(curr > size) curr = size;
    else { if(curr < 1) {curr = 1;}}

    /*
    std::cout << " HCALProperties::eta2ieta  end of iter." << iter 
	      << " curr, etatow[curr-1], etatow[curr] = "
	      << curr << " " << etatow[curr-1] << " " << etatow[curr] << std::endl;
    */
    
  }

  /*
  std::cout << " HCALProperties::eta2ieta  for input x = " << x 
	    << "  found index = " << curr-1
	    << std::endl;
  */
  
  return curr-1;
}
