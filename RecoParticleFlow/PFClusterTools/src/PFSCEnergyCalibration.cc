#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

using namespace std;

PFSCEnergyCalibration::PFSCEnergyCalibration() {}


PFSCEnergyCalibration::PFSCEnergyCalibration(std::vector<double>& barrelFbremCorr,
					     std::vector<double>& endcapFbremCorr,
					     std::vector<double>& barrelCorr,
					     std::vector<double>& endcapCorr):
  barrelFbremCorr_(barrelFbremCorr),
  endcapFbremCorr_(endcapFbremCorr),
  barrelCorr_(barrelCorr),
  endcapCorr_(endcapCorr)
{

  // intial parameters

  bool log = false;
  
  if(barrelFbremCorr_.size() != 13)
    edm::LogError("PFSCEnergyCalibration")<<" wrong size input paramter: calibPFSCEle_Fbrem_barrel read = "
					  << barrelCorr_.size() << " expected = 13" << endl;
  
  
  if(endcapFbremCorr_.size() != 13)
    edm::LogError("PFSCEnergyCalibration")<<" wrong size input parameter: calibPFSCEle_Fbrem_endcap read = "
					  << endcapCorr_.size() << " expected = 13" << endl;

  
  
  if(barrelCorr_.size() != 17)
    edm::LogError("PFSCEnergyCalibration")<<" wrong size input paramter: calibPFSCEle_barrel read = "
					  << barrelCorr_.size() << " expected = 17" << endl;
  
  
  if(endcapCorr_.size() != 9)
    edm::LogError("PFSCEnergyCalibration")<<" wrong size input parameter: calibPFSCEle_endcap read = "
					  << endcapCorr_.size() << " expected = 9" << endl;

  
  if(log)
    cout << " ****** THE BARREL SC FBREM CORRECTIONS ******* " << barrelFbremCorr_.size()  << endl;
  for(unsigned int ip = 0; ip< barrelFbremCorr_.size(); ip++){
    pbb[ip] = barrelFbremCorr_[ip];
    if(log)
      cout << " pbb[" << ip << "] " << " = " << pbb[ip] << endl;
  }
  
  if(log)
    cout << " ****** THE ENCCAP SC FBREM CORRECTIONS ******* " << endcapFbremCorr_.size() << endl;
  for(unsigned int ip = 0; ip< endcapFbremCorr_.size(); ip++){
    pbe[ip] = endcapFbremCorr_[ip];
    if(log)
      cout << " pbe[" << ip << "] " << " = " << pbe[ip] << endl;
  }
  
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


double PFSCEnergyCalibration::SCCorrFBremBarrel(double e, double et, double brLinear) { //MM
  double fCorr = 1;

  // make NO correction if brLinear is invalid!
  if ( brLinear == 0 ) return e;
  //
  
  if ( brLinear < pbb[0] ) brLinear = pbb[0];
  if ( brLinear > pbb[1] ) brLinear = pbb[1];
  
  
  double p0 = pbb[2]; 
  double p1 = pbb[3];  
  double p2 = pbb[4]; 
  double p3 = pbb[5];
  double p4 = pbb[6];  
  

  //Low pt ( < 25 GeV) dedicated corrections
  if( et < pbb[7] ) {
    p0 = pbb[8]; 
    p1 = pbb[9];  
    p2 = pbb[10];  
    p3 = pbb[11];
    p4 = pbb[12];
  }

  double threshold = p4;
  
  double y = p0*threshold*threshold + p1*threshold + p2;
  double yprime = 2*p0*threshold + p1;
  double a = p3;
  double b = yprime - 2*a*threshold;
  double c = y - a*threshold*threshold - b*threshold;
  
  if ( brLinear < threshold ) 
    fCorr = p0*brLinear*brLinear + p1*brLinear + p2;
  else 
    fCorr = a*brLinear*brLinear + b*brLinear + c;
  
  return e/fCorr;

}


double PFSCEnergyCalibration::SCCorrFBremEndcap(double e, double eta, double brLinear) {//MM
  double fCorr = 1;

  //Energy must contain associated preshower energy

  if ( brLinear == 0 ) return e;
  
  if ( brLinear < pbe[0] ) brLinear = pbe[0];
  if ( brLinear > pbe[1] ) brLinear = pbe[1];
  
  
  double p0 = pbe[2]; 
  double p1 = pbe[3];  
  double p2 = pbe[4]; 
  double p3 = pbe[5];
  double p4 = pbe[6];  

  //Change of set of corrections to take
  //into account the active preshower region
  if(fabs(eta) > pbe[7] ) {
    p0 = pbe[8]; 
    p1 = pbe[9];  
    p2 = pbe[10];  
    p3 = pbe[11];
    p4 = pbe[12];   
  }

  double threshold = p4;
    
  double y = p0*threshold*threshold + p1*threshold + p2;
  double yprime = 2*p0*threshold + p1;
  double a = p3;
  double b = yprime - 2*a*threshold;
  double c = y - a*threshold*threshold - b*threshold;

  if ( brLinear < threshold ) 
    fCorr = p0*brLinear*brLinear + p1*brLinear + p2;
  else 
    fCorr = a*brLinear*brLinear + b*brLinear + c;
    
  return e/fCorr;


}


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
  
  //2010 corrections
//  double temp_et = et;
//   // Avoid energy correction divergency at low Et. 
//   if(temp_et < 2)
//     temp_et = 2;
  
//   double d0 = 15.0;     // sharpness of the curve
//   double d1 = -0.00181;
//   double d2 = 1.081;
  
//   double p0 = bb[0] + bb[1]/(temp_et + bb[2]) - bb[3]/(temp_et) ;
//   double p1 = bb[4] + bb[5]/(bb[6] + temp_et);

  

//   // for the momentum the fixed value d2 is prefered to p2
//   double p2 = bb[7] + bb[8]*temp_et + bb[9]*temp_et*temp_et + bb[10]*temp_et*temp_et*temp_et;
  
//   if(temp_et > 130) {
//     double y = 130;
//     p2 = bb[7] + bb[8]*y + bb[9]*y*y + bb[10]*y*y*y;
//   }
   
//   fCorr = p0 + p1*atan(d0*(d2 - fabs(eta))) + d1*fabs(eta);
 

  //February 2011 corrections
  double temp_et = et;
  // Avoid energy correction divergency at low Et. 
  if(temp_et < 3)
    temp_et = 3;

  double c0 = bb[0];
  double c1 = bb[1];
  double c2 = bb[2];
  double c3 = bb[3];

  double d0 = bb[4];
  double d1 = bb[5];
  double d2 = bb[6];
  double d3 = bb[7];

  double e0 = 1.081;  // curve point in eta distribution
  double e1 = 7.6;     // sharpness of the curve
  double e2 = -0.00181;

  //Low pt ( < 25 GeV) dedidacted corrections
  if(temp_et < bb[8] ) {

    c0 = bb[9];
    c1 = bb[10];
    c2 = bb[11];
    c3 = bb[12];
    
    d0 = bb[13];
    d1 = bb[14];
    d2 = bb[15];
    d3 = bb[16];
    
  }
  
  double p0 = c0 + c1/(temp_et + c2) + c3/(temp_et*temp_et);
  double p1 = d0/(temp_et + d1) + d2/(temp_et*temp_et + d3);


  fCorr = p0 + p1*atan(e1*(e0 - fabs(eta))) + e2*fabs(eta);
  
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
 
  double temp_et = et;
  // Avoid energy correction divergency at low Et. 
  if(temp_et < 3)
    temp_et = 3;

  double p0 = cc[0] + cc[1]/(cc[2] + temp_et);
  double p1 = cc[3] + cc[4]/(cc[5] + temp_et);
  double p2 = cc[6] + cc[7]/(cc[8] + temp_et);
    
    
    fCorr = p0 + p1*fabs(eta) +  p2*eta*eta;
    
    return et/fCorr;
}

