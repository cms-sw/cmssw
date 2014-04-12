#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkApvMask.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkNoiseCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkPedestalCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonModeCalculator.h"
#include <algorithm>

using namespace std;
ApvAnalysis::ApvAnalysis(int nEvForUpdate)
{

  theTkCommonModeCalculator =0;
  theTkPedestalCalculator =0;
  theTkNoiseCalculator =0;
  theTkApvMask =0;
  nEventsForNoiseCalibration_ =0;
  eventsRequiredToUpdate_ = nEvForUpdate;



}
void ApvAnalysis::newEvent() const{
  theTkPedestalCalculator->newEvent();
  theTkNoiseCalculator->newEvent();
  theTkCommonModeCalculator->newEvent();
}

void ApvAnalysis::updateCalibration(edm::DetSet<SiStripRawDigi>& in) {
  theTkPedestalCalculator->updatePedestal(in);
  
  PedestalType noise;
  if(theTkPedestalCalculator->status()->isUpdating()){
    nEventsForNoiseCalibration_++; 

    if(theTkNoiseCalculator->noise().size() == 0) {
      noise = theTkPedestalCalculator->rawNoise();
      theTkNoiseCalculator->setStripNoise(noise);
      theTkApvMask->calculateMask(noise);
    }

    PedestalType pedestal= theTkPedestalCalculator->pedestal();
    PedestalType tmp;
    tmp.clear();
    edm::DetSet<SiStripRawDigi>::const_iterator it = in.data.begin();
    int i=0;
    for(;it!= in.data.end();it++){
      tmp.push_back((*it).adc() - pedestal[i]);
      i++;
    }
    PedestalType tmp2 = theTkCommonModeCalculator->doIt(tmp);
    if(tmp2.size() > 0) {
      theTkNoiseCalculator->updateNoise(tmp2);
    }   
    if(nEventsForNoiseCalibration_%eventsRequiredToUpdate_ == 1 && nEventsForNoiseCalibration_ >1)
      {

	noise=theTkNoiseCalculator->noise();
	theTkApvMask->calculateMask(noise);
	
      }
  }


}
