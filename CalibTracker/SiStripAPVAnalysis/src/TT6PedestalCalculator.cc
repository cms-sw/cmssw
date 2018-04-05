#include "CalibTracker/SiStripAPVAnalysis/interface/TT6PedestalCalculator.h"

#include <cmath>
#include <numeric>
#include <algorithm>
using namespace std;
TT6PedestalCalculator::TT6PedestalCalculator(int evnt_ini, 
                        int evnt_iter, float sig_cut) :   
                        numberOfEvents(0),
                        alreadyUsedEvent(false)
{
  if (false) cout << "Constructing TT6PedestalCalculator " << endl;
  eventsRequiredToCalibrate = evnt_ini; 
  eventsRequiredToUpdate    = evnt_iter;
  cutToAvoidSignal          = sig_cut;
  init();
}
//
// Initialization.
//
void TT6PedestalCalculator::init() { 
  theRawNoise.clear();
  thePedestal.clear();
  thePedSum.clear();
  thePedSqSum.clear();
  theEventPerStrip.clear();
  theStatus.setCalibrating();
}
//
//  -- Destructor  
//
TT6PedestalCalculator::~TT6PedestalCalculator() {
  if (false) cout << "Destructing TT6PedestalCalculator " << endl;
}
//
// -- Set Pedestal Update Status
//
void TT6PedestalCalculator::updateStatus(){

  if (theStatus.isCalibrating() && 
      numberOfEvents >= eventsRequiredToCalibrate) {
    theStatus.setUpdating();
  }
}
//
// -- Initialize or Update (when needed) Pedestal Values
//
void TT6PedestalCalculator::updatePedestal(ApvAnalysis::RawSignalType& in) {
  if (alreadyUsedEvent == false) {
    alreadyUsedEvent = true;
    numberOfEvents++;
    if (theStatus.isCalibrating()) {
      initializePedestal(in);
    } else if (theStatus.isUpdating()) {
      refinePedestal(in);
    }  
    updateStatus();
  }
}
//
// -- Initialize Pedestal Values using a set of events (eventsRequiredToCalibrate)
//
void TT6PedestalCalculator::initializePedestal(ApvAnalysis::RawSignalType& in) {
  if (numberOfEvents == 1) {

    thePedSum.clear();
    thePedSqSum.clear();
    theEventPerStrip.clear();
    
    thePedSum.reserve(128);
    thePedSqSum.reserve(128);
    theEventPerStrip.reserve(128);
    
    thePedSum.resize(in.data.size(), 0.0);
    thePedSqSum.resize(in.data.size(), 0.0);
    theEventPerStrip.resize(in.data.size(), 0);
  }
  if (numberOfEvents <= eventsRequiredToCalibrate) {
    edm::DetSet<SiStripRawDigi>::const_iterator i = in.data.begin();
    int ii=0;
    for (;i!=in.data.end() ; i++) {
      thePedSum[ii]   += (*i).adc();
      thePedSqSum[ii] += ((*i).adc())*((*i).adc());
      theEventPerStrip[ii]++;
      ii++;
    }
  }
  if (numberOfEvents == eventsRequiredToCalibrate) {
    thePedestal.clear();
    theRawNoise.clear();
    edm::DetSet<SiStripRawDigi>::const_iterator i = in.data.begin();
    int ii=0;
    for (;i!=in.data.end() ; i++) {
      double avVal   = (theEventPerStrip[ii])	? thePedSum[ii]/theEventPerStrip[ii]:0.0;
      double sqAvVal = (theEventPerStrip[ii])	? thePedSqSum[ii]/theEventPerStrip[ii]:0.0;
      double corr_fac = (theEventPerStrip[ii] > 1) ? (theEventPerStrip[ii]/(theEventPerStrip[ii]-1)) : 1.0;
      double rmsVal  =  (sqAvVal - avVal*avVal > 0.0) ? sqrt(corr_fac * (sqAvVal - avVal*avVal)) : 0.0;	

      thePedestal.push_back(static_cast<float>(avVal));
      theRawNoise.push_back(static_cast<float>(rmsVal));
      ii++;
    }
  }
}
//
// -- Update Pedestal Values when needed.
//
void TT6PedestalCalculator::refinePedestal(ApvAnalysis::RawSignalType& in) {
  if (((numberOfEvents-eventsRequiredToCalibrate)%eventsRequiredToUpdate) == 1) {
    
    thePedSum.clear();
    thePedSqSum.clear();
    theEventPerStrip.clear();
    
    thePedSum.reserve(128);
    thePedSqSum.reserve(128);
    theEventPerStrip.reserve(128);
    
    thePedSum.resize(in.data.size(), 0.0);
    thePedSqSum.resize(in.data.size(), 0.0);
    theEventPerStrip.resize(in.data.size(), 0);
  }
  unsigned int ii=0;
  ApvAnalysis::RawSignalType::const_iterator i= in.data.begin();
  for (; i < in.data.end(); i++) {
    if (fabs((*i).adc()-thePedestal[ii]) < cutToAvoidSignal*theRawNoise[ii]) {
      thePedSum[ii]   += (*i).adc();
      thePedSqSum[ii] += ((*i).adc())*((*i).adc());
      theEventPerStrip[ii]++;
    }
    ii++;
  }
  if (((numberOfEvents-eventsRequiredToCalibrate) % eventsRequiredToUpdate) == 0) {
    
    for (unsigned int iii = 0; iii < in.data.size(); iii++) {
      if (theEventPerStrip[iii] > 10 ) {
        double avVal   = (theEventPerStrip[iii]) ? thePedSum[iii]/theEventPerStrip[iii]:0.0;
        double sqAvVal = (theEventPerStrip[iii]) ? thePedSqSum[iii]/theEventPerStrip[iii]:0.0;
        double rmsVal  =  (sqAvVal - avVal*avVal > 0.0) ? sqrt(sqAvVal - avVal*avVal) : 0.0;	

	
        if (avVal != 0 ) {
          thePedestal[iii] = static_cast<float>(avVal);
          theRawNoise[iii] = static_cast<float>(rmsVal);
        }
      }
    }
    thePedSum.clear();
    thePedSqSum.clear();
    theEventPerStrip.clear();    
        }
      }
      //
      // Define New Event
// 
void TT6PedestalCalculator::newEvent(){
  alreadyUsedEvent = false;
}
