#ifndef SiPixelPerformanceSummary_h
#define SiPixelPerformanceSummary_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <iostream>
#include <cstdint>

#define kDetSummarySize 60  // float numbers kept in DetSummary.performanceValues
#define kDefaultValue -99.9

class SiPixelPerformanceSummary {
public:
  struct DetSummary {
    uint32_t detId_;
    std::vector<float> performanceValues_;

    COND_SERIALIZABLE;
  };

  class StrictWeakOrdering {  // sort detSummaries by detId
  public:
    bool operator()(const DetSummary& detSumm, const uint32_t& otherDetId) const {
      return (detSumm.detId_ < otherDetId);
    };
  };

public:
  SiPixelPerformanceSummary(const SiPixelPerformanceSummary&);
  SiPixelPerformanceSummary();
  ~SiPixelPerformanceSummary();

  void clear();

  unsigned int size() { return allDetSummaries_.size(); }

  void setTimeStamp(unsigned long long timeStamp) { timeStamp_ = timeStamp; }
  unsigned long long getTimeStamp() const { return timeStamp_; }

  void setRunNumber(unsigned int runNumber) { runNumber_ = runNumber; }
  unsigned int getRunNumber() const { return runNumber_; }

  void setNumberOfEvents(unsigned int numberOfEvents) { numberOfEvents_ = numberOfEvents; }
  unsigned int getNumberOfEvents() const { return numberOfEvents_; }

  void setLuminosityBlock(unsigned int lumBlock) { luminosityBlock_ = lumBlock; }
  unsigned int getLuminosityBlock() const { return luminosityBlock_; };

  void print() const;
  void print(const uint32_t detId) const;
  void printAll() const;

  std::vector<uint32_t> getAllDetIds() const;
  std::vector<DetSummary> getAllDetSummaries() const { return allDetSummaries_; }
  std::vector<float> getDetSummary(uint32_t detId) const;

  // RawData
  bool setRawDataErrorType(uint32_t detId, int bin, float nErrors);
  // Digi
  bool setNumberOfDigis(uint32_t detId, float mean, float rms, float emPtn);
  bool setADC(uint32_t detId, float mean, float rms, float emPtn);
  // Cluster
  bool setNumberOfClusters(uint32_t detId, float mean, float rms, float emPtn);
  bool setClusterCharge(uint32_t detId, float mean, float rms, float emPtn);
  bool setClusterSize(uint32_t detId, float mean, float rms, float emPtn);
  bool setClusterSizeX(uint32_t detId, float mean, float rms, float emPtn);
  bool setClusterSizeY(uint32_t detId, float mean, float rms, float emPtn);
  // RecHit
  bool setNumberOfRecHits(uint32_t detId, float mean, float rms, float emPtn);
  // TrackResidual
  bool setResidualX(uint32_t detId, float mean, float rms, float emPtn);
  bool setResidualY(uint32_t detId, float mean, float rms, float emPtn);
  //
  bool setNumberOfNoisCells(uint32_t detId, float nNpixCells);  // N=4,1..
  bool setNumberOfDeadCells(uint32_t detId, float nNpixCells);  // N=4,1..
  bool setNumberOfPixelHitsInTrackFit(uint32_t detId, float nPixelHits);
  // Track
  bool setFractionOfTracks(uint32_t detId, float mean, float rms);
  bool setNumberOfOnTrackClusters(uint32_t detId, float nClusters);
  bool setNumberOfOffTrackClusters(uint32_t detId, float nClusters);
  bool setClusterChargeOnTrack(uint32_t detId, float mean, float rms);
  bool setClusterChargeOffTrack(uint32_t detId, float mean, float rms);
  bool setClusterSizeOnTrack(uint32_t detId, float mean, float rms);
  bool setClusterSizeOffTrack(uint32_t detId, float mean, float rms);

private:
  std::pair<bool, std::vector<DetSummary>::iterator> initDet(const uint32_t detId);
  std::pair<bool, std::vector<DetSummary>::iterator> setDet(const uint32_t detId,
                                                            const std::vector<float>& performanceValues);
  bool setValue(uint32_t detId, int index, float performanceValue);
  float getValue(uint32_t detId, int index);

private:
  unsigned long long timeStamp_;
  unsigned int runNumber_;
  unsigned int luminosityBlock_;
  unsigned int numberOfEvents_;

  std::vector<DetSummary> allDetSummaries_;

  COND_SERIALIZABLE;
};

#endif
