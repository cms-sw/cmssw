#include "CalibTracker/SiStripAPVAnalysis/interface/MedianCommonModeCalculator.h"
#include <cmath>

using namespace std;
MedianCommonModeCalculator::MedianCommonModeCalculator() : 
  //  theNoiseCalculator(noise_calc),
    //  theApvMask(mask_calc),
  alreadyUsedEvent(false)
{
  if (0) cout << "Constructing MedianCommonMode Calculator ..." << endl;
  //  cutToAvoidSignal = sig_cut;
}
//
//  Destructor 
//
MedianCommonModeCalculator::~MedianCommonModeCalculator() {
  if (0) cout << "Destructing TT6CommonModeCalculator " << endl;
}
//
// Action :
//
ApvAnalysis::PedestalType MedianCommonModeCalculator::doIt
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
void MedianCommonModeCalculator::calculateCommonMode(ApvAnalysis::PedestalType& indat) 
{ 
  if (alreadyUsedEvent == false) {
    alreadyUsedEvent = true;
    
    theCommonModeValues.clear();
    
    
    double avVal = 0.0;
    
    sort(indat.begin(),indat.end());
	
    uint16_t index = indat.size()%2 ? indat.size()/2 : indat.size()/2-1;
    if ( !indat.empty() ) { avVal = indat[index]; }
    
    theCommonModeValues.push_back(static_cast<float>(avVal));
       
    MedianCommonModeCalculator::setCM(theCommonModeValues);

  }
  
}

//
// Define New Event
// 
void MedianCommonModeCalculator::newEvent() {
  alreadyUsedEvent = false;
}
