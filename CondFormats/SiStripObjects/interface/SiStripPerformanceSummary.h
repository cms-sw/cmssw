#ifndef SiStripPerformanceSummary_h
#define SiStripPerformanceSummary_h
#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>
// -*- C++ -*-
// Package:    CondFormats/SiStripObjects
// Class:      SiStripPerformanceSummary
/**\class SiStripPerformanceSummary SiStripPerformanceSummary.h CondFormats/SiStripObjects/SiStripPerformanceSummary.h
 Description: <one line class summary>
 Implementation:
     <Object for writing to a DB the long-term detector performance of the Silicon Strip Tracker>
*/
// Original Author:  Dorian Kcira
//         Created:  Mon Apr 30 17:43:00 CEST 2007
// $Id$

#define kDetSummarySize 7    // the number of values than can be kept in DetSummary.performanceValues
#define kApvSummarySize 4    // number of values than can be kept in ApvSummary.performanceValues
#define kNonsenseValue -199. // value to be used as a non meaningful result
class SiStripPerformanceSummary {
 public:
   // structure for keeping data
   struct DetSummary{
     uint32_t detId;
     std::vector<float> performanceValues;
   };
   // ordering DetSummary-s according to detid
   class StrictWeakOrdering{
   public:
     bool operator() (const DetSummary& dSumm, const uint32_t& otherDetId) const {return dSumm.detId < otherDetId;};
   };
   // matching a DetSummary just by Detid
   class MatchDetSummaryDetId : public std::unary_function<DetSummary, bool> {
   public:
     MatchDetSummaryDetId(const uint32_t& input_detid): detid_(input_detid){ } // keep memory of the input detid
     bool operator() (const DetSummary& dSumm) const {return dSumm.detId == detid_;} // compare to that input detid
   private:
     uint32_t detid_;
   };
 public:
   // public methods
   SiStripPerformanceSummary();
   SiStripPerformanceSummary(const SiStripPerformanceSummary&);
   ~SiStripPerformanceSummary();
   void clear(){vDetSummary_.clear();}; // reset the list of summaries
   unsigned int size(){return vDetSummary_.size();}; // size of list of summaries
   // general methods
   void print() const; // general print out
   void print(const uint32_t detid) const; // print summary for specific detid
   void printall() const; // print all summaries
   void print(const std::vector<float>& pvec) const; // print a vector of floats
   void getSummary(const uint32_t input_detid, std::vector<float>& voutput) const;
   void getDetIds(std::vector<uint32_t>& vdetids) const;
   // methods for setting concrete values
/*
vector<float> performanceValues[kDetSummarySize];
  0. mean   Cluster Size
  1. RMS    Cluster Size
  2. Mean   Cluster Charge
  3. RMS    Cluster Charge
  4. Mean   Occupancy
  5. RMS    Occupancy
  6. % Number of Noisy Strips
*/
   bool setClusterSize(uint32_t input_detid, float clusterSizeMean, float clusterSizeRMS){
     return setTwoValues(input_detid,clusterSizeMean,clusterSizeRMS,0,1);
   };
   bool setClusterCharge(uint32_t input_detid, float clusterChargeMean, float clusterChargeRMS){
     return setTwoValues(input_detid,clusterChargeMean,clusterChargeRMS,2,3);
   };
   bool setOccupancy(uint32_t input_detid, float occupancyMean, float occupancyRMS){
     return setTwoValues(input_detid,occupancyMean,occupancyRMS,4,5);
   };
   bool setPercentNoisyStrips(uint32_t input_detid, float noisyStrips){
     return setOneValue(input_detid,noisyStrips,6);
   };
   void setRunNr(unsigned int inputRunNr){ runNr_ = inputRunNr;};
   unsigned int getRunNr() const {return runNr_;};
   void setTimeValue(unsigned long long inputTimeValue){timeValue_=inputTimeValue;};
   unsigned long long getTimeValue() const {return timeValue_;};
   std::vector<DetSummary> getWholeSummary() const {return vDetSummary_;};
   // make private methods below?
   std::pair<bool, std::vector<DetSummary>::iterator> setDet(const uint32_t input_detid, const std::vector<float>& input_values);
   std::pair<bool, std::vector<DetSummary>::iterator> initDet(const uint32_t input_detid); // set to default values
 private:
   bool setTwoValues(uint32_t input_detid, float val1, float val2, int index1, int index2);
   bool setOneValue(uint32_t input_detid, float val1, int index1);
   float getOneValue(uint32_t input_detid, int index1);
 private: // data members
   std::vector<DetSummary> vDetSummary_;
   unsigned int runNr_;
   unsigned long long timeValue_;
};
#endif
