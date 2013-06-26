#include "CalibTracker/SiStripAPVAnalysis/interface/TT6CommonModeCalculator.h"
#include <cmath>

using namespace std;
TT6CommonModeCalculator::TT6CommonModeCalculator(TkNoiseCalculator* noise_calc, TkApvMask* mask_calc, float sig_cut) : 
  theNoiseCalculator(noise_calc),
  theApvMask(mask_calc),
  alreadyUsedEvent(false)
{
  if (0) cout << "Constructing TT6CommonMode Calculator ..." << endl;
  cutToAvoidSignal = sig_cut;
}
//
//  Destructor 
//
TT6CommonModeCalculator::~TT6CommonModeCalculator() {
  if (0) cout << "Destructing TT6CommonModeCalculator " << endl;
}
//
// Action :
//
ApvAnalysis::PedestalType TT6CommonModeCalculator::doIt
                (const ApvAnalysis::PedestalType& _indat) 
{
  ApvAnalysis::PedestalType indat = _indat;
  ApvAnalysis::PedestalType out;
  calculateCommonMode(indat);
  int setNumber;
  if(theCommonModeValues.size() >0) {
    for (unsigned int i=0; i<indat.size(); i++){
      setNumber = theTkCommonMode->topology().setOfStrip(i);
      out.push_back(indat[i] - theCommonModeValues[setNumber]);
    }  
  }else{
    out = indat;
  }
  return out;
}
//
//  Calculation of Common Mode Values :
//
void TT6CommonModeCalculator::calculateCommonMode(ApvAnalysis::PedestalType& indat) 
{ 
  if (alreadyUsedEvent == false) {
    alreadyUsedEvent = true;
    //  cout<< "I am inside the calculateCommonMode"<<endl;
    TkApvMask::MaskType strip_mask = theApvMask->mask();
    ApvAnalysis::PedestalType strip_noise = theNoiseCalculator->noise();
    theCommonModeValues.clear();
    
    if(strip_noise.size() > 0) {
      int nSet = theTkCommonMode->topology().numberOfSets();
      for (int i=0; i<nSet; i++){
        int initial   = theTkCommonMode->topology().initialStrips()[i];
        int final     = theTkCommonMode->topology().finalStrips()[i];
        double sumVal = 0.0;
        double sumWt =  0.0;
        for (int j = initial; j <= final; j++) {
          if (strip_mask[j] == TkApvMask::ok ) {
            if(fabs(indat[j]) < cutToAvoidSignal*strip_noise[j]) { 
              double nWeight = 1/(strip_noise[j]*strip_noise[j]);
              sumVal += (indat[j]*nWeight);
              sumWt += nWeight;
            }
          }
        }
        double avVal = (sumWt) ? sumVal/sumWt :0.0;
        theCommonModeValues.push_back(static_cast<float>(avVal));
        //cout <<"Setting CM values"<<endl;
      }
    }
  }
  TT6CommonModeCalculator::setCM(theCommonModeValues);
  calculateCMSlope(indat);     
}
//
// Define New Event
// 
void TT6CommonModeCalculator::newEvent() {
  alreadyUsedEvent = false;
}
//
// Calculate CMSlope 
// 
void TT6CommonModeCalculator::calculateCMSlope(ApvAnalysis::PedestalType& indat) {
  if (indat.size() != 128) {
    slope = -100.0;
    return;
  }
  ApvAnalysis::PedestalType diffVec;
  diffVec.clear();
  for(int s=0;s<64;s++) diffVec.push_back(indat[s+64]-indat[s]);
  std::sort(diffVec.begin(),diffVec.end());
  slope = (diffVec[31]+diffVec[32])/2./64.;
}

