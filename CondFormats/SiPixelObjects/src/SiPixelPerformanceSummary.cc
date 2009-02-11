#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"


SiPixelPerformanceSummary::SiPixelPerformanceSummary() : runNumber_(0), timeValue_(0) {}


SiPixelPerformanceSummary::SiPixelPerformanceSummary(const SiPixelPerformanceSummary& spps) {
  runNumber_ = spps.getRunNumber();
  timeValue_ = spps.getTimeValue();
  allDetSummaries_ = spps.getAllDetSummaries();
}


SiPixelPerformanceSummary::~SiPixelPerformanceSummary() {}


std::pair<bool, std::vector<SiPixelPerformanceSummary::DetSummary>::iterator> 
                SiPixelPerformanceSummary::initDet(const uint32_t detId) { 
  std::vector<float> performanceValues; 
  for (int i=0; i<kDetSummarySize; ++i) performanceValues.push_back(kDefaultValue);
  return setDet(detId, performanceValues);
}


std::pair<bool, std::vector<SiPixelPerformanceSummary::DetSummary>::iterator> 
                SiPixelPerformanceSummary::setDet(const uint32_t detId, 
		                                  const std::vector<float>& performanceValues) {
  std::vector<DetSummary>::iterator iDetSum = allDetSummaries_.end();
  
  if (performanceValues.size()!=kDetSummarySize) { // for inappropriate input
    edm::LogError("Error") << "wrong input size = "<< performanceValues.size() 
                   	   <<" can only add "<< kDetSummarySize 
		    	   <<" values; NOT adding to SiPixelPerformanceSummary";
    return std::make_pair(false, iDetSum);
  }
  iDetSum = std::lower_bound(allDetSummaries_.begin(), allDetSummaries_.end(), 
                             detId, SiPixelPerformanceSummary::StrictWeakOrdering());

  if (iDetSum!=allDetSummaries_.end() && // for an existong entry 
      iDetSum->detId_==detId) return std::make_pair(false, iDetSum); 

  
  DetSummary detSummary; // for a new entry, put at (position-1) returned by StrictWeakOrdering(?) 
  	     detSummary.detId_ = detId;
  	     detSummary.performanceValues_ = performanceValues;
  return std::make_pair(true, allDetSummaries_.insert(iDetSum, detSummary));
}


bool SiPixelPerformanceSummary::setValue(uint32_t detId, int index, float performanceValue) {
  if (index>kDetSummarySize) {
    edm::LogError("SetError") << "could not set the performance value for index = "<< index <<" > "<< kDetSummarySize;
    return false;
  }
  std::pair<bool, std::vector<DetSummary>::iterator> initResult = initDet(detId);
  if (initResult.first==true || initResult.second!=allDetSummaries_.end()) { 
    initResult.second->performanceValues_[index] = performanceValue;
    return true;
  }
  else {
    edm::LogError("SetError") << "could not set the performance value; " 
                              << "new entry could not be created for detId = "<< detId;
    return false; 
  }
  return true;
}


float SiPixelPerformanceSummary::getValue(uint32_t detId, int index) {
  if (index>kDetSummarySize) {
    edm::LogError("GetError") << "could not get values for index = "<< index <<" > "<< kDetSummarySize;
    return kDefaultValue;
  }
  std::vector<float> performanceValues; 
  performanceValues.clear();
  getDetSummary(detId, performanceValues);
  if (performanceValues.size()==kDetSummarySize) return performanceValues[index]; 
  else return kDefaultValue; 
}


bool SiPixelPerformanceSummary::setNumberOfDigis(uint32_t detId, float mean, float RMS) {
  bool mSet = setValue(detId, 0, mean);
  bool rSet = setValue(detId, 1, RMS);
  return (mSet && rSet); 
}


bool SiPixelPerformanceSummary::setNoisePercentage(uint32_t detId, float percentage) {
  return setValue(detId, 2, percentage);
}


void SiPixelPerformanceSummary::getAllDetIds(std::vector<uint32_t>& allDetIds) const {
  std::vector<DetSummary>::const_iterator begin = allDetSummaries_.begin();
  std::vector<DetSummary>::const_iterator end = allDetSummaries_.end();
  for (std::vector<DetSummary>::const_iterator iDetSum=begin; 
       iDetSum!=end; ++iDetSum) allDetIds.push_back(iDetSum->detId_);
}


void SiPixelPerformanceSummary::getDetSummary(const uint32_t detId, std::vector<float>& performanceValues) const {
  std::vector<DetSummary>::const_iterator iDetSum = std::find_if(allDetSummaries_.begin(), 
                                                                 allDetSummaries_.end(), 
							         MatchDetSummaryDetId(detId));
  if (iDetSum==allDetSummaries_.end()) edm::LogError("get") << "cannot find any detSummary for detId = "<< detId;
  else performanceValues = iDetSum->performanceValues_; 
}


void SiPixelPerformanceSummary::print() const {
  edm::LogInfo("print") << "SiPixelPerformanceSummary size = "<< allDetSummaries_.size() 
                        << "; run number = "<< runNumber_ << "; time value = "<< timeValue_ << std::endl;
}


void SiPixelPerformanceSummary::print(const uint32_t detId) const {
  std::vector<DetSummary>::const_iterator iDetSum = std::find_if(allDetSummaries_.begin(), 
                                                                 allDetSummaries_.end(), 
								 MatchDetSummaryDetId(detId));
  if (iDetSum==allDetSummaries_.end()) edm::LogError("print") << "cannot find any DetSummary for detId = "<< detId; 
  else {
    edm::LogInfo("print") << "detId = "<< detId <<" detSummary for detId = "<< iDetSum->detId_;
    print(iDetSum->performanceValues_);
  }
}


void SiPixelPerformanceSummary::print(const std::vector<float>& performanceValues) const {
 for (std::vector<float>::const_iterator iPV=performanceValues.begin(); 
                                         iPV!=performanceValues.end(); ++iPV) std::cout <<" "<< *iPV;
 std::cout << std::endl;
}


void SiPixelPerformanceSummary::printall() const {
  print();
  std::vector<DetSummary>::const_iterator begin = allDetSummaries_.begin();
  std::vector<DetSummary>::const_iterator end = allDetSummaries_.end();
  for (std::vector<DetSummary>::const_iterator iDetSum=begin; iDetSum!=end; ++iDetSum) {
    std::cout <<" detId = "<< iDetSum->detId_;
    print(iDetSum->performanceValues_);
  }
}
