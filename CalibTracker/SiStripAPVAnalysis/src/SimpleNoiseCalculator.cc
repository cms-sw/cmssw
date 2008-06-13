#include "CalibTracker/SiStripAPVAnalysis/interface/SimpleNoiseCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>
#include <numeric>
#include <algorithm>
using namespace std;
//
//  Constructors
//
SimpleNoiseCalculator::SimpleNoiseCalculator() :
  numberOfEvents(0) ,
  alreadyUsedEvent(false)  
{
  if (0) cout << "Constructing SimpleNoiseCalculator " << endl;
  init();
}
//
SimpleNoiseCalculator::SimpleNoiseCalculator(int evnt_ini, bool use_DB) :
  numberOfEvents(0) ,
  alreadyUsedEvent(false)  
{
  if (0) cout << "Constructing SimpleNoiseCalculator " << endl;
  useDB_ = use_DB;
  eventsRequiredToCalibrate_ = evnt_ini;
  //  eventsRequiredToUpdate_    = evnt_iter;
  //  cutToAvoidSignal_          = sig_cut;
  init();
}
//
// Initialization.
//
void SimpleNoiseCalculator::init() {
  theCMPSubtractedSignal.clear();
  theNoise.clear();
  theNoiseSum.clear();
  theNoiseSqSum.clear();
  theEventPerStrip.clear();
  // theStatus.setCalibrating();
}
//
//  Destructor 
//
SimpleNoiseCalculator::~SimpleNoiseCalculator() {
  if (0) cout << "Destructing SimpleNoiseCalculator " << endl;
}
//
// Update the Status of Noise Calculation
//
void SimpleNoiseCalculator::updateStatus(){
  if ( (theStatus.isCalibrating() && numberOfEvents >= eventsRequiredToCalibrate_) || (useDB_==true && numberOfEvents ==1) ) {
    theStatus.setUpdating();
  }
}
//
// Calculate and update (when needed) Noise Values
//
void SimpleNoiseCalculator::updateNoise(ApvAnalysis::PedestalType& in){
  if (alreadyUsedEvent == false) {
    alreadyUsedEvent = true;
    numberOfEvents++;

    if (numberOfEvents == 1 && theNoise.size() != in.size()) {
      edm::LogError("SimpleNoiseCalculator:updateNoise") << " You did not initialize the Noise correctly prior to noise calibration.";
    }

    // Initialize sums used for estimating noise.
    if (numberOfEvents == 1)
    {
      theNoiseSum.clear();
      theNoiseSqSum.clear();
      theEventPerStrip.clear();

      theNoiseSum.resize(in.size(),0.0);
      theNoiseSqSum.resize(in.size(),0.0);
      theEventPerStrip.resize(in.size(),0);
    }    

    unsigned int i;

    // At every event Update sums used for estimating noise.
    for (i = 0; i < in.size(); i++) {

        theNoiseSum[i]   += in[i];
        theNoiseSqSum[i] += in[i]*in[i];
        theEventPerStrip[i]++;
    }

    // Calculate noise.
    if ((theStatus.isCalibrating() && numberOfEvents == eventsRequiredToCalibrate_) || theStatus.isUpdating() )
    {
      theCMPSubtractedSignal.clear();
      theNoise.clear();

      for (i = 0; i < in.size(); i++) {
        double avVal   = (theEventPerStrip[i]) 
          ? theNoiseSum[i]/(theEventPerStrip[i]):0.0;
        double sqAvVal = (theEventPerStrip[i]) 
          ? theNoiseSqSum[i]/(theEventPerStrip[i]):0.0;
        double corr_fac = (theEventPerStrip[i] > 1) 
          ? (theEventPerStrip[i]/(theEventPerStrip[i]-1)) : 1.0;
        double rmsVal  =  (sqAvVal - avVal*avVal > 0.0) 
          ? sqrt(corr_fac * (sqAvVal - avVal*avVal)) : 0.0;	
      
        theCMPSubtractedSignal.push_back(static_cast<float>(avVal));

        theNoise.push_back(static_cast<float>(rmsVal));
  
        if (0) cout << " SimpleNoiseCalculator::updateNoise " 
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
void SimpleNoiseCalculator::newEvent() {
  alreadyUsedEvent = false;
}


