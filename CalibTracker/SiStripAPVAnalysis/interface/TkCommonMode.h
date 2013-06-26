#ifndef TkCommonMode_H
#define TkCommonMode_H

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonModeTopology.h"

#include <vector>
/**
 * A common mode class which can work with any common mode topology,
 * where the topology refers to the number of strips for which a
 * common mode value is calculed (128 or less). Currently quite slow....
 */
class TkCommonMode {
 public:
  virtual ~TkCommonMode(){}

  virtual TkCommonModeTopology& topology() {return *myTkCommonModeTopology;}
  virtual void setTopology(TkCommonModeTopology* in) {myTkCommonModeTopology = in;}
  
  /** Set the independent CM values in the APV */
  void setCommonMode(const std::vector<float>& in) {theCommonMode = in;}
  /** Return vector containing all the independent CM values in the APV. */
  std::vector<float> returnAsVector() const {return theCommonMode;}
  /** Return vector of dimension 128, with CM value on each strip */
  std::vector<float> toVector() const; // This return a full vector, with duplicates...
 private:
  TkCommonModeTopology* myTkCommonModeTopology;
  std::vector<float> theCommonMode;
};

#endif
