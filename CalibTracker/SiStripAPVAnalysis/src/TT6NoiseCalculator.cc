#include "CalibTracker/SiStripAPVAnalysis/interface/TT6NoiseCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>
#include <numeric>
#include <algorithm>
using namespace std;
//
//  Constructors
//
TT6NoiseCalculator::TT6NoiseCalculator() :
  numberOfEvents(0) ,
  alreadyUsedEvent(false)  
{
  if (0) cout << "Constructing TT6NoiseCalculator " << endl;
  init();
}
//
TT6NoiseCalculator::TT6NoiseCalculator(int evnt_ini,
                   int evnt_iter, float sig_cut) :
  numberOfEvents(0) ,
  alreadyUsedEvent(false)  
{
  if (0) cout << "Constructing TT6NoiseCalculator " << endl;
  eventsRequiredToCalibrate_ = evnt_ini;
  eventsRequiredToUpdate_    = evnt_iter;
  cutToAvoidSignal_          = sig_cut;
  init();
}
//
// Initialization.
//
void TT6NoiseCalculator::init() {
  theCMPSubtractedSignal.clear();
  theNoise.clear();
  theNoiseSum.clear();
  theNoiseSqSum.clear();
  theEventPerStrip.clear();
  theStatus.setCalibrating();
}
//
//  Destructor 
//
TT6NoiseCalculator::~TT6NoiseCalculator() {
  if (0) cout << "Destructing TT6NoiseCalculator " << endl;
}
//
// Update the Status of Noise Calculation
//
void TT6NoiseCalculator::updateStatus(){
  if (theStatus.isCalibrating() && 
      numberOfEvents >= eventsRequiredToCalibrate_) {
    theStatus.setUpdating();
  }
}
//
// Calculate and update (when needed) Noise Values
//
void TT6NoiseCalculator::updateNoise(ApvAnalysis::PedestalType& in){
  if (alreadyUsedEvent == false) {
    alreadyUsedEvent = true;
    numberOfEvents++;

    if (numberOfEvents == 1 && theNoise.size() != in.size()) {
      edm::LogError("TT6NoiseCalculator:updateNoise") << " You did not initialize the Noise correctly prior to noise calibration.";
    }

    // Initialize sums used for estimating noise.
    if ((theStatus.isCalibrating() && numberOfEvents == 1) ||
        (theStatus.isUpdating() && (numberOfEvents - eventsRequiredToCalibrate_)%eventsRequiredToUpdate_ == 1)) 
    {
      theNoiseSum.clear();
      theNoiseSqSum.clear();
      theEventPerStrip.clear();

      theNoiseSum.resize(in.size(),0.0);
      theNoiseSqSum.resize(in.size(),0.0);
      theEventPerStrip.resize(in.size(),0);
    }    

    unsigned int i;

    // Update sums used for estimating noise.
    for (i = 0; i < in.size(); i++) {
      if (fabs(in[i]) < cutToAvoidSignal_*theNoise[i]) {
        theNoiseSum[i]   += in[i];
        theNoiseSqSum[i] += in[i]*in[i];
        theEventPerStrip[i]++;
      }
    }

    // Calculate noise.
    if ((theStatus.isCalibrating() && numberOfEvents == eventsRequiredToCalibrate_) ||
        (theStatus.isUpdating() && (numberOfEvents - eventsRequiredToCalibrate_)%eventsRequiredToUpdate_ == 0))
    {
      theCMPSubtractedSignal.clear();
      theNoise.clear();

      for (i = 0; i < in.size(); i++) {
        double avVal   = (theEventPerStrip[i]) ? theNoiseSum[i]/(theEventPerStrip[i]):0.0;
        double sqAvVal = (theEventPerStrip[i]) ? theNoiseSqSum[i]/(theEventPerStrip[i]):0.0;
        double corr_fac = (theEventPerStrip[i] > 1) ? (theEventPerStrip[i]/(theEventPerStrip[i]-1)) : 1.0;
        double rmsVal  =  (sqAvVal - avVal*avVal > 0.0) ? sqrt(corr_fac * (sqAvVal - avVal*avVal)) : 0.0;	
      
        theCMPSubtractedSignal.push_back(static_cast<float>(avVal));

        theNoise.push_back(static_cast<float>(rmsVal));
  
        if (0) cout << " TT6NoiseCalculator::updateNoise " 
                    << theNoiseSum[i] << " " 
                    << theNoiseSqSum[i] << " "
                    << theEventPerStrip[i] << " "  
                    << avVal << " " 
                    << sqAvVal << " " 
                    << (sqAvVal - avVal*avVal) << " " 
                    << rmsVal << endl;
      }
    }
    updateStatus();
  }
}
//
// Define New Event
// 
void TT6NoiseCalculator::newEvent() {
  alreadyUsedEvent = false;
}


