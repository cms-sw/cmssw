#include "CalibTracker/SiStripAPVAnalysis/interface/TT6ApvMask.h"
#include <cmath>
#include <numeric>
#include <algorithm>
using namespace std;
// 
//  Constructors:
//
TT6ApvMask::TT6ApvMask(int ctype, float ncut, float dcut, float tcut) {
  theCalculationFlag_ = ctype;
  theNoiseCut_        = ncut;
  theDeadCut_         = dcut;
  theTruncationCut_   = tcut;
}

// 
//  Destructor :
//
TT6ApvMask::~TT6ApvMask(){
  if (0) cout << "Destructing TT6ApvMask " << endl;
}
//
// Calculate the Mask 
//
void TT6ApvMask::calculateMask(const ApvAnalysis::PedestalType& in){

  theMask_.clear();
  ApvAnalysis::PedestalType temp_in(in);
  double sumVal,sqSumVal,avVal,sqAvVal,rmsVal; 
  sort(temp_in.begin(), temp_in.end());
  int nSize    = in.size();
  int cutLow   = int(nSize * theTruncationCut_);
  int cutHigh  = int(nSize * theTruncationCut_);
  int effSize  = nSize - cutLow - cutHigh;
  sumVal = 0.0;
  sqSumVal = 0.0;
  sumVal   = accumulate((temp_in.begin()+cutLow), (temp_in.end()-cutHigh), 0.0);
  sqSumVal = inner_product((temp_in.begin()+cutLow), (temp_in.end()-cutHigh), 
                           (temp_in.begin()+cutLow), 0.0);
  
  avVal    = (effSize) ? sumVal/float(effSize):0.0;
  sqAvVal  = (effSize) ? sqSumVal/float(effSize):0.0;
  rmsVal   = (sqAvVal - avVal*avVal > 0.0) ? sqrt(sqAvVal - avVal*avVal):0.0; 
  if (0) cout << " TT6ApvMask::calculateMask  Mean " << avVal <<
	   " RMS " << rmsVal << " " <<  effSize << endl;       
  for (unsigned int i=0; i<in.size(); i++){
    if (defineNoisy( static_cast<float>(avVal),
                     static_cast<float>(rmsVal),in[i])) {
      theMask_.push_back(noisy);
    } else if (in[i] < theDeadCut_*avVal) {
           theMask_.push_back(dead);
    } else {
      theMask_.push_back(ok);
    }	  
  }
}
//
//  Identification of Noisy strips (three options available : using cut on 
//   rms of noice distribution, using a percentage cut wrt the average, using
//   a fixed cut 
//
bool TT6ApvMask::defineNoisy(float avrg,float rms,float noise){
  bool temp;
  temp=false;
  if (theCalculationFlag_ == 1){
   if ((noise-avrg) > theNoiseCut_*rms) {
     temp=true;
     if (0) cout << " Mean " << avrg << " rms " << rms << " Noise " << noise << endl;
   }
  } else if (theCalculationFlag_ == 2){
    if ((noise-avrg) > avrg*theNoiseCut_) temp=true;
  } else if (theCalculationFlag_ == 3){
    if (noise > theNoiseCut_) temp=true;
  } 
  return temp;
}
