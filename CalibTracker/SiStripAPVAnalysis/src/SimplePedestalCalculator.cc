#include "CalibTracker/SiStripAPVAnalysis/interface/SimplePedestalCalculator.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;
SimplePedestalCalculator::SimplePedestalCalculator(int evnt_ini) :   
                        numberOfEvents(0),
                        alreadyUsedEvent(false)
{
  if (0) cout << "Constructing SimplePedestalCalculator " << endl;
  eventsRequiredToCalibrate = evnt_ini; 
  //  eventsRequiredToUpdate    = evnt_iter;
  //  cutToAvoidSignal          = sig_cut;
  init();
}
//
// Initialization.
//
void SimplePedestalCalculator::init() { 
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
SimplePedestalCalculator::~SimplePedestalCalculator() {
  if (0) cout << "Destructing SimplePedestalCalculator " << endl;
}


//
// -- Set Pedestal Update Status
//
void SimplePedestalCalculator::updateStatus(){
  if (theStatus.isCalibrating() && 
      numberOfEvents >= eventsRequiredToCalibrate) {
    theStatus.setUpdating();
  }
}


//
// -- Initialize or Update (when needed) Pedestal Values
//
void SimplePedestalCalculator::updatePedestal(ApvAnalysis::RawSignalType& in) {

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
void SimplePedestalCalculator::initializePedestal(ApvAnalysis::RawSignalType& in) {
  if (numberOfEvents == 1) {

    thePedSum.clear();
    thePedSqSum.clear();
    theEventPerStrip.clear();
    
    thePedSum.reserve(128);
    thePedSqSum.reserve(128);
    theEventPerStrip.reserve(128);
    
    thePedSum.resize(in.data.size(), 0);
    thePedSqSum.resize(in.data.size(), 0);
    theEventPerStrip.resize(in.data.size(),0);
  }

  //eventsRequiredToCalibrate is considered the minimum number of events to be used

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


      // the pedestal is calculated as int, as required by FED.
      int avVal   = (theEventPerStrip[ii]) 
	? thePedSum[ii]/theEventPerStrip[ii]:0;

      double sqAvVal = (theEventPerStrip[ii]) 
	? thePedSqSum[ii]/theEventPerStrip[ii]:0.0;
      double corr_fac = (theEventPerStrip[ii] > 1) 
	? (theEventPerStrip[ii]/(theEventPerStrip[ii]-1)) : 1.0;
      double rmsVal  =  (sqAvVal - avVal*avVal > 0.0) 
           ? sqrt(corr_fac * (sqAvVal - avVal*avVal)) : 0.0;	
      thePedestal.push_back(static_cast<float>(avVal));
      theRawNoise.push_back(static_cast<float>(rmsVal));
      ii++;
    }

  }
}

//
// -- Update Pedestal Values when needed.
//

void SimplePedestalCalculator::refinePedestal(ApvAnalysis::RawSignalType& in) {


  // keep adding th adc count for any events 

  unsigned int ii=0;
  ApvAnalysis::RawSignalType::const_iterator i= in.data.begin();
  for (; i < in.data.end(); i++) {
    
    thePedSum[ii]   += (*i).adc();
    thePedSqSum[ii] += ((*i).adc())*((*i).adc());
    theEventPerStrip[ii]++;
    
    ii++;
  }


  // calculate a new pedestal any events, so it will come for free when for the last event

  for (unsigned int iii = 0; iii < in.data.size(); iii++) {
    if (theEventPerStrip[iii] > 10 ) {
      int avVal   = (theEventPerStrip[iii]) 
	? thePedSum[iii]/theEventPerStrip[iii]:0;

      double sqAvVal = (theEventPerStrip[iii]) 
	? thePedSqSum[iii]/theEventPerStrip[iii]:0.0;

      double rmsVal  =  (sqAvVal - avVal*avVal > 0.0) 
	? sqrt(sqAvVal - avVal*avVal) : 0.0;	
      
      
      if (avVal != 0 ) {
	thePedestal[iii] = static_cast<float>(avVal);
	theRawNoise[iii] = static_cast<float>(rmsVal);
      }
    }
  }

}





//
// Define New Event
// 

void SimplePedestalCalculator::newEvent(){
  alreadyUsedEvent = false;
}
