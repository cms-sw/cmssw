#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"


SiPixelPerformanceSummary::SiPixelPerformanceSummary() : runNumber_(0), 
                                                         numberOfEvents_(0), 
							 timeValue_(0) {}


SiPixelPerformanceSummary::SiPixelPerformanceSummary(const SiPixelPerformanceSummary& spps) {
  runNumber_ = spps.getRunNumber();
  numberOfEvents_ = spps.getNumberOfEvents(); 
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

// for SiPixelMonitorRawData: 

bool SiPixelPerformanceSummary::setNumberOfRawDataErrors(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 0, mean) && setValue(detId, 1, rms)); 
} 

bool SiPixelPerformanceSummary::setRawDataErrorType(uint32_t detId, int bin, float percentage) {
  return setValue(detId, 2+bin, percentage);
}

bool SiPixelPerformanceSummary::setTBMType(uint32_t detId, int bin, float percentage) {
  return setValue(detId, 16+bin, percentage);
}

bool SiPixelPerformanceSummary::setTBMMessage(uint32_t detId, int bin, float percentage) {
  return setValue(detId, 21+bin, percentage);
}

bool SiPixelPerformanceSummary::setFEDfullType(uint32_t detId, int bin, float percentage) {
  return setValue(detId, 29+bin, percentage);
}

bool SiPixelPerformanceSummary::setFEDtimeoutChannel(uint32_t detId, int bin, float percentage) {
  return setValue(detId, 36+bin, percentage);
}

bool SiPixelPerformanceSummary::setSLinkErrSize(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 73, mean) && setValue(detId, 74, rms)); 
}
 
bool SiPixelPerformanceSummary::setFEDmaxErrLink(uint32_t detId, float maxErrID) {
  return setValue(detId, 75, maxErrID);
}
 
bool SiPixelPerformanceSummary::setmaxErr36ROC(uint32_t detId, float maxErrID) {
  return setValue(detId, 76, maxErrID);
}
 
bool SiPixelPerformanceSummary::setmaxErrDCol(uint32_t detId, float maxErrID) {
  return setValue(detId, 77, maxErrID);
}
 
bool SiPixelPerformanceSummary::setmaxErrPixelRow(uint32_t detId, float maxErrID) {
  return setValue(detId, 78, maxErrID);
}
 
bool SiPixelPerformanceSummary::setmaxErr38ROC(uint32_t detId, float maxErrID) {
  return setValue(detId, 79, maxErrID);
}
 
// for SiPixelMonitorDigi

bool SiPixelPerformanceSummary::setNumberOfDigis(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 80, mean) && setValue(detId, 81, rms)); 
}

bool SiPixelPerformanceSummary::setADC(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 82, mean) && setValue(detId, 83, rms)); 
}

bool SiPixelPerformanceSummary::setDigimapHotCold(uint32_t detId, float hot, float cold) {
  return (setValue(detId, 84, hot) && setValue(detId, 85, cold)); 
}

// for SiPixelMonitorCluster

bool SiPixelPerformanceSummary::setNumberOfClusters(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 86, mean) && setValue(detId, 87, rms)); 
}

bool SiPixelPerformanceSummary::setClusterCharge(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 88, mean) && setValue(detId, 89, rms)); 
}

bool SiPixelPerformanceSummary::setClusterSizeX(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 90, mean) && setValue(detId, 91, rms)); 
}

bool SiPixelPerformanceSummary::setClusterSizeY(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 92, mean) && setValue(detId, 93, rms)); 
}

bool SiPixelPerformanceSummary::setClustermapHotCold(uint32_t detId, float hot, float cold) {
  return (setValue(detId, 94, hot) && setValue(detId, 95, cold)); 
}

// for SiPixelMonitorRecHit: 

bool SiPixelPerformanceSummary::setNumberOfRecHits(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 96, mean) && setValue(detId, 97, rms)); 
}


bool SiPixelPerformanceSummary::setRecHitMatchedClusterSizeX(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 98, mean) && setValue(detId, 99, rms)); 
}

bool SiPixelPerformanceSummary::setRecHitMatchedClusterSizeY(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 100, mean) && setValue(detId, 101, rms)); 
}

bool SiPixelPerformanceSummary::setRecHitmapHotCold(uint32_t detId, float hot, float cold) {
  return (setValue(detId, 102, hot) && setValue(detId, 103, cold)); 
}

// for SiPixelMonitorTrack(Residual): 

bool SiPixelPerformanceSummary::setResidualX(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 104, mean) && setValue(detId, 105, rms)); 
}

bool SiPixelPerformanceSummary::setResidualY(uint32_t detId, float mean, float rms) {
  return (setValue(detId, 106, mean) && setValue(detId, 107, rms)); 
}


void SiPixelPerformanceSummary::getAllDetIds(std::vector<uint32_t>& allDetIds) const {
  std::vector<DetSummary>::const_iterator begin = allDetSummaries_.begin();
  std::vector<DetSummary>::const_iterator end = allDetSummaries_.end();
  for (std::vector<DetSummary>::const_iterator iDetSum=begin; 
       iDetSum!=end; ++iDetSum) allDetIds.push_back(iDetSum->detId_);
}


void SiPixelPerformanceSummary::getDetSummary(const uint32_t detId, 
                                              std::vector<float>& performanceValues) const {
  std::vector<DetSummary>::const_iterator iDetSum = std::find_if(allDetSummaries_.begin(), 
                                                                 allDetSummaries_.end(), 
							         MatchDetSummaryDetId(detId));
  if (iDetSum==allDetSummaries_.end()) edm::LogError("get") << "cannot find any detSummary for detId = "
                                                            <<  detId;
  else performanceValues = iDetSum->performanceValues_; 
}


void SiPixelPerformanceSummary::print() const {
  edm::LogInfo("print") << "SiPixelPerformanceSummary size = "<< allDetSummaries_.size() 
                        << "; run number = "<< runNumber_ 
                        << "; number of events = "<< numberOfEvents_ 
			<< "; time value = "<< timeValue_ << std::endl;
}


void SiPixelPerformanceSummary::print(const uint32_t detId) const {
  std::vector<DetSummary>::const_iterator iDetSum = std::find_if(allDetSummaries_.begin(), 
                                                                 allDetSummaries_.end(), 
								 MatchDetSummaryDetId(detId));
  if (iDetSum==allDetSummaries_.end()) edm::LogError("print") << "cannot find any DetSummary for detId = "
                                                              <<  detId; 
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


void SiPixelPerformanceSummary::printAll() const {
  print();
  std::vector<DetSummary>::const_iterator begin = allDetSummaries_.begin();
  std::vector<DetSummary>::const_iterator end = allDetSummaries_.end();
  for (std::vector<DetSummary>::const_iterator iDetSum=begin; iDetSum!=end; ++iDetSum) {
    std::cout <<" detId = "<< iDetSum->detId_;
    print(iDetSum->performanceValues_);
  }
}
