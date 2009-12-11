#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

using namespace std;

PFSCEnergyCalibration::PFSCEnergyCalibration() {}


PFSCEnergyCalibration::PFSCEnergyCalibration(std::vector<double>& barrelCorr,
					     std::vector<double>& endcapCorr):
  barrelCorr_(barrelCorr),
  endcapCorr_(endcapCorr)
{

  // intial parameters

  bool log = false;
  
  
  if(barrelCorr_.size() != 11)
    edm::LogError("PFSCEnergyCalibration")<<" wrong size input paramter: calibPFSCEle_barrel read = "
					  << barrelCorr_.size() << " expected = 11" << endl;
  
  
  if(endcapCorr_.size() != 9)
    edm::LogError("PFSCEnergyCalibration")<<" wrong size input parameter: calibPFSCEle_endcap read = "
				     << endcapCorr_.size() << " expected = 8" << endl;

  
  if(log)
    cout << " ****** THE BARREL SC CORRECTIONS ******* " << barrelCorr_.size()  << endl;
  for(unsigned int ip = 0; ip< barrelCorr_.size(); ip++){
    bb[ip] = barrelCorr_[ip];
    if(log)
      cout << " bb[" << ip << "] " << " = " << bb[ip] << endl;
  }
  
  if(log)
    cout << " ****** THE ENCCAP SC CORRECTIONS ******* " << endcapCorr_.size() << endl;
  for(unsigned int ip = 0; ip< endcapCorr_.size(); ip++){
    cc[ip] = endcapCorr_[ip];
    if(log)
      cout << " cc[" << ip << "] " << " = " << cc[ip] << endl;
  }
}
PFSCEnergyCalibration::~PFSCEnergyCalibration() {}


double PFSCEnergyCalibration::SCCorrEtEtaBarrel(double et, double eta) {
  double fCorr = 0;
  
  
  // 25 November Morning
  
//   //p0  
//   double bb0 = 1.03257;
//   double bb1 = -1.37103e+01;
//   double bb2 = 3.39716e+02;
//   double bb3 = 4.86192e-01;
  
//   //p1
//   double bb4 = 1.81653e-03;
//   double bb5 = 3.64445e-01;
//   double bb6 = 1.41132;
  
  
//   //p2
//   double bb7 = 1.02061;
//   double bb8 = 5.91624e-03;
//   double bb9 = -5.14434e-05;
//   double bb10 = 1.42516e-07; 
  


  
  double d0 = 15.0;     // sharpness of the curve
  double d1 = -0.00181;
  double d2 = 1.081;
  
  double p0 = bb[0] + bb[1]/(et + bb[2]) - bb[3]/(et) ;
  double p1 = bb[4] + bb[5]/(bb[6] + et);

  

  // for the momentum the fixed value d2 is prefered to p2
  double p2 = bb[7] + bb[8]*et + bb[9]*et*et + bb[10]*et*et*et;
  
  if(et > 130) {
    double y = 130;
    p2 = bb[7] + bb[8]*y + bb[9]*y*y + bb[10]*y*y*y;
  }
  



  
  fCorr = p0 + p1*atan(d0*(d2 - fabs(eta))) + d1*fabs(eta);
  
  return et/fCorr;
}

double PFSCEnergyCalibration::SCCorrEtEtaEndcap(double et, double eta) {
  double fCorr = 0;
  

//   //p0
//   double c0 = 9.99464e-01;
//   double c1 = -1.23130e+01;
//   double c2 = 2.87841;
  
//   //p1
//   double c3 = -1.05697e-04;
//   double c4 = 1.02819e+01;
//   double c5 = 3.05904; 
  
  
//   //p2
//   double c6 = 1.35017e-03;
//   double c7 = -2.21845;
//   double c8 = 3.42062;
  


  double p0 = cc[0] + cc[1]/(cc[2] + et);
  double p1 = cc[3] + cc[4]/(cc[5] + et);
  double p2 = cc[6] + cc[7]/(cc[8] + et);
    
    
    fCorr = p0 + p1*fabs(eta) +  p2*eta*eta;
    
    return et/fCorr;
}

