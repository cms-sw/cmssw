#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"

#include <algorithm>

using namespace edm; 
using namespace std; 


SiPixelPerformanceSummary::SiPixelPerformanceSummary() 
                         : timeStamp_(0), runNumber_(0), luminosityBlock_(0), numberOfEvents_(0) {}


SiPixelPerformanceSummary::SiPixelPerformanceSummary(const SiPixelPerformanceSummary& performanceSummary) {
        timeStamp_ = performanceSummary.getTimeStamp();
        runNumber_ = performanceSummary.getRunNumber();
  luminosityBlock_ = performanceSummary.getLuminosityBlock();
   numberOfEvents_ = performanceSummary.getNumberOfEvents(); 
  allDetSummaries_ = performanceSummary.getAllDetSummaries();
}


SiPixelPerformanceSummary::~SiPixelPerformanceSummary() {}


void SiPixelPerformanceSummary::clear() {
  timeStamp_=0; runNumber_=0; luminosityBlock_=0; numberOfEvents_=0; allDetSummaries_.clear(); 
}


pair<bool, vector<SiPixelPerformanceSummary::DetSummary>::iterator> 
                  SiPixelPerformanceSummary::initDet(const uint32_t detId) { 
  vector<float> performanceValues; 
  for (int i=0; i<kDetSummarySize; ++i) performanceValues.push_back(kDefaultValue);
  return setDet(detId, performanceValues);
}


pair<bool, vector<SiPixelPerformanceSummary::DetSummary>::iterator> 
                  SiPixelPerformanceSummary::setDet(const uint32_t detId, 
		                                    const vector<float>& performanceValues) {
  vector<DetSummary>::iterator iDetSumm = allDetSummaries_.end();
  
  if (performanceValues.size()!=kDetSummarySize) { // for inappropriate input
    cout << "not adding these "<< performanceValues.size() << " values; " 
         << "SiPixelPerformanceSummary can only add "<< kDetSummarySize <<" values per DetSummary";
    return make_pair(false, iDetSumm);
  }
  iDetSumm = lower_bound(allDetSummaries_.begin(), allDetSummaries_.end(), 
                         detId, SiPixelPerformanceSummary::StrictWeakOrdering()); 
  
  if (iDetSumm!=allDetSummaries_.end() && // for an existong entry 
      iDetSumm->detId_==detId) return make_pair(false, iDetSumm); 
  
  DetSummary newDetSumm; // for a new entry, put at (position-1) returned by StrictWeakOrdering
  	     newDetSumm.detId_ = detId; 
	     newDetSumm.performanceValues_ = performanceValues;
  return make_pair(true, allDetSummaries_.insert(iDetSumm, newDetSumm));
}


bool SiPixelPerformanceSummary::setValue(uint32_t detId, int index, float performanceValue) {
  if (index>kDetSummarySize) {
    cout << "cannot set the performance value for index = "<< index <<" > "<< kDetSummarySize;
    return false;
  }
  pair<bool, vector<DetSummary>::iterator> initResult = initDet(detId);
  if (initResult.first || initResult.second!=allDetSummaries_.end()) { 
    initResult.second->performanceValues_[index] = performanceValue;
    return true;
  }
  else {
    cout << "cannot set the performance value; cannot create new entry for detId = "<< detId;
    return false; 
  }
  return true;
}


float SiPixelPerformanceSummary::getValue(uint32_t detId, int index) {
  if (index>kDetSummarySize) {
    cout << "cannot get value for detId = "<< detId <<" index = "<< index <<" > "<< kDetSummarySize; 
    return kDefaultValue;
  }
  vector<float> performanceValues = getDetSummary(detId);
  if (performanceValues.size()==kDetSummarySize) return performanceValues[index]; 
  else return kDefaultValue; 
}


bool SiPixelPerformanceSummary::setRawDataErrorType(uint32_t detId, int bin, float nErrors) {
  return  setValue(detId, bin, nErrors);
}


bool SiPixelPerformanceSummary::setNumberOfDigis(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 15, mean) && setValue(detId, 16, rms) && setValue(detId, 17, emPtn)); 
}

bool SiPixelPerformanceSummary::setADC(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 18, mean) && setValue(detId, 19, rms) && setValue(detId, 20, emPtn)); 
}


bool SiPixelPerformanceSummary::setNumberOfClusters(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 21, mean) && setValue(detId, 22, rms) && setValue(detId, 23, emPtn)); 
}

bool SiPixelPerformanceSummary::setClusterCharge(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 24, mean) && setValue(detId, 25, rms) && setValue(detId, 26, emPtn)); 
}

bool SiPixelPerformanceSummary::setClusterSize(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 27, mean) && setValue(detId, 28, rms) && setValue(detId, 29, emPtn)); 
}

bool SiPixelPerformanceSummary::setClusterSizeX(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 30, mean) && setValue(detId, 31, rms) && setValue(detId, 32, emPtn)); 
}

bool SiPixelPerformanceSummary::setClusterSizeY(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 33, mean) && setValue(detId, 34, rms) && setValue(detId, 35, emPtn)); 
}


bool SiPixelPerformanceSummary::setNumberOfRecHits(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 36, mean) && setValue(detId, 37, rms) && setValue(detId, 38, emPtn)); 
}


bool SiPixelPerformanceSummary::setResidualX(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 39, mean) && setValue(detId, 40, rms) && setValue(detId, 41, emPtn)); 
}

bool SiPixelPerformanceSummary::setResidualY(uint32_t detId, float mean, float rms, float emPtn) {
  return (setValue(detId, 42, mean) && setValue(detId, 43, rms) && setValue(detId, 44, emPtn)); 
}


bool SiPixelPerformanceSummary::setNumberOfNoisCells(uint32_t detId, float nNpixCells) {
  return  setValue(detId, 45, nNpixCells); 
}

bool SiPixelPerformanceSummary::setNumberOfDeadCells(uint32_t detId, float nNpixCells) {
  return  setValue(detId, 46, nNpixCells); 
}

bool SiPixelPerformanceSummary::setNumberOfPixelHitsInTrackFit(uint32_t detId, float nPixelHits) {
  return  setValue(detId, 47, nPixelHits); 
}

bool SiPixelPerformanceSummary::setFractionOfTracks(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 48, mean) && setValue(detId, 49, rms)); 
}

bool SiPixelPerformanceSummary::setNumberOfOnTrackClusters(uint32_t detId, float nClusters) {
  return setValue(detId, 50, nClusters); 
}

bool SiPixelPerformanceSummary::setNumberOfOffTrackClusters(uint32_t detId, float nClusters) {
  return setValue(detId, 51, nClusters); 
}

bool SiPixelPerformanceSummary::setClusterChargeOnTrack(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 52, mean) && setValue(detId, 53, rms)); 
}

bool SiPixelPerformanceSummary::setClusterChargeOffTrack(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 54, mean) && setValue(detId, 55, rms)); 
}

bool SiPixelPerformanceSummary::setClusterSizeOnTrack(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 56, mean) && setValue(detId, 57, rms)); 
}

bool SiPixelPerformanceSummary::setClusterSizeOffTrack(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 58, mean) && setValue(detId, 59, rms)); 
}

vector<uint32_t> SiPixelPerformanceSummary::getAllDetIds() const {
  vector<uint32_t> allDetIds; 
  for (vector<DetSummary>::const_iterator iDetSumm = allDetSummaries_.begin(); 
       iDetSumm!=allDetSummaries_.end(); ++iDetSumm) allDetIds.push_back(iDetSumm->detId_); 
  return allDetIds; 
}


vector<float> SiPixelPerformanceSummary::getDetSummary(const uint32_t detId) const {
  vector<DetSummary>::const_iterator iDetSumm = find_if(allDetSummaries_.begin(), 
                                                        allDetSummaries_.end(), 
                                                        [&detId](const DetSummary& detSumm) -> bool
                                                                {return detSumm.detId_ == detId;}
                                                        );
  if (iDetSumm==allDetSummaries_.end()) { 
    vector<float> performanceValues; 
    cout << "cannot get DetSummary for detId = "<< detId; 
    return performanceValues; 
  }
  else return iDetSumm->performanceValues_; 
}


void SiPixelPerformanceSummary::print(const uint32_t detId) const {
  vector<float> performanceValues = getDetSummary(detId);   
  cout << "DetSummary for detId "<< detId <<" : ";
  for (vector<float>::const_iterator v = performanceValues.begin(); v!=performanceValues.end(); ++v) cout <<" "<< *v; 
  cout << endl;
}


void SiPixelPerformanceSummary::print() const {
  cout << "SiPixelPerformanceSummary size (allDets) = "<< allDetSummaries_.size() << ", "
       << "time stamp = "<< timeStamp_ << ", "
       << "run number = "<< runNumber_ << ", "
       << "luminosity section = "<< luminosityBlock_ << ", "
       << "number of events = "<< numberOfEvents_ << endl;
}


void SiPixelPerformanceSummary::printAll() const {
  print();
  for (vector<DetSummary>::const_iterator iDetSumm = allDetSummaries_.begin(); 
       iDetSumm!=allDetSummaries_.end(); ++iDetSumm) print(iDetSumm->detId_); 
}
