#ifndef SiPixelPerformanceSummary_h
#define SiPixelPerformanceSummary_h


#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


#define kDetSummarySize 108 // number of values kept in DetSummary.performanceValues
#define kDefaultValue -99.9 


class SiPixelPerformanceSummary {
public:
  struct DetSummary { 
    uint32_t detId_;
    std::vector<float> performanceValues_;
  }; 
  class StrictWeakOrdering { // sort DetSummaries by detId
  public:
    bool operator() (const DetSummary& detSumm, const uint32_t& otherDetId) const { 
      return (detSumm.detId_ < otherDetId); 
    };
  };
  class MatchDetSummaryDetId : public std::unary_function<DetSummary, bool> { 
  public: 
    MatchDetSummaryDetId(const uint32_t& detId) : detId_(detId) {} 
    bool operator() (const DetSummary& detSumm) const { return (detSumm.detId_==detId_); } 
  private:
    uint32_t detId_;
  };
  
public:
  SiPixelPerformanceSummary(const SiPixelPerformanceSummary&);
  SiPixelPerformanceSummary();
 ~SiPixelPerformanceSummary();
  
  void clear() { allDetSummaries_.clear(); }; 
  unsigned int size() { return allDetSummaries_.size(); };

  std::vector<DetSummary> getAllDetSummaries() const { return allDetSummaries_; };

          void setRunNumber(unsigned int runNumber) { runNumber_ = runNumber; }; 
  unsigned int getRunNumber() const { return runNumber_; };
  
          void setNumberOfEvents(unsigned int numberOfEvents) { numberOfEvents_ = numberOfEvents; }; 
  unsigned int getNumberOfEvents() const { return numberOfEvents_; };
  
                void setTimeValue(unsigned long long timeValue) { timeValue_ = timeValue; };
  unsigned long long getTimeValue() const { return timeValue_; };
    
  void print() const; 
  void print(const uint32_t detId) const; 
  void print(const std::vector<float>& performanceValues) const; 
  void printAll() const;  
  
  std::pair<bool, std::vector<DetSummary>::iterator> initDet(const uint32_t detId); 
  std::pair<bool, std::vector<DetSummary>::iterator>  setDet(const uint32_t detId, 
                                                             const std::vector<float>& performanceValues); 
  void getAllDetIds(std::vector<uint32_t>& vDetIds) const;
  void getDetSummary(const uint32_t detId, std::vector<float>& performanceValues) const;

// RawData 
  bool setNumberOfRawDataErrors(uint32_t detId, float mean, float rms); 
  bool setRawDataErrorType(uint32_t detId, int bin, float percentage);
  bool setTBMType(uint32_t detId, int bin, float percentage);
  bool setTBMMessage(uint32_t detId, int bin, float percentage);
  bool setFEDfullType(uint32_t detId, int bin, float percentage); 
  bool setFEDtimeoutChannel(uint32_t detId, int bin, float percentage); 
  bool setSLinkErrSize(uint32_t detId, float mean, float rms); 
  bool setFEDmaxErrLink(uint32_t detId, float maxErrID); 
  bool setmaxErr36ROC(uint32_t detId, float maxErrID); 
  bool setmaxErrDCol(uint32_t detId, float maxErrID); 
  bool setmaxErrPixelRow(uint32_t detId, float maxErrID); 
  bool setmaxErr38ROC(uint32_t detId, float maxErrID); 
// Digi
  bool setNumberOfDigis(uint32_t detId, float mean, float rms);
  bool setADC(uint32_t detId, float mean, float rms);
  bool setDigimapHotCold(uint32_t detId, float hot, float cold);
// Cluster
  bool setNumberOfClusters(uint32_t detId, float mean, float rms);
  bool setClusterCharge(uint32_t detId, float mean, float rms); 
  bool setClusterSizeX(uint32_t detId, float mean, float rms);
  bool setClusterSizeY(uint32_t detId, float mean, float rms);
  bool setClustermapHotCold(uint32_t detId, float hot, float cold);
// RecHit
  bool setNumberOfRecHits(uint32_t detId, float mean, float rms);
  bool setRecHitMatchedClusterSizeX(uint32_t detId, float mean, float rms);
  bool setRecHitMatchedClusterSizeY(uint32_t detId, float mean, float rms);
  bool setRecHitmapHotCold(uint32_t detId, float hot, float cold);
// TrackResidual: 
  bool setResidualX(uint32_t detId, float mean, float rms);
  bool setResidualY(uint32_t detId, float mean, float rms);

private:
   bool setValue(uint32_t detid, int index, float performanceValue);
  float getValue(uint32_t detid, int index);

private: 
  std::vector<DetSummary> allDetSummaries_;
  unsigned int runNumber_;
  unsigned int numberOfEvents_;
  unsigned long long timeValue_;
};


#endif
