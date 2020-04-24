#ifndef CondFormats_SiStripObjects_DaqScopeModeAnalysis_H
#define CondFormats_SiStripObjects_DaqScopeModeAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

/**
   @class DaqScopeModeAnalysis
   @author R.Bainbridge
   @brief Analysis for scope mode data.
*/

class DaqScopeModeAnalysis : public CommissioningAnalysis {
  
 public:
  
  DaqScopeModeAnalysis( const uint32_t& key );

  DaqScopeModeAnalysis();

  ~DaqScopeModeAnalysis() override {;}

  friend class DaqScopeModeAlgorithm;
  
  inline const float& entries() const;

  inline const float& mean() const; 

  inline const float& median() const; 

  inline const float& mode() const; 

  inline const float& rms() const; 

  inline const float& min() const; 

  inline const float& max() const; 
  
  void print( std::stringstream&, uint32_t not_used = 0 ) override;
  
  void reset() override;
  
 private:
  
  float entries_;

  float mean_;

  float median_;

  float mode_;

  float rms_;

  float min_;

  float max_;
  
};

const float& DaqScopeModeAnalysis::entries() const { return entries_; }
const float& DaqScopeModeAnalysis::mean() const { return mean_; }
const float& DaqScopeModeAnalysis::median() const { return median_; }
const float& DaqScopeModeAnalysis::mode() const { return mode_; }
const float& DaqScopeModeAnalysis::rms() const { return rms_; }
const float& DaqScopeModeAnalysis::min() const { return min_; }
const float& DaqScopeModeAnalysis::max() const { return max_; }

#endif // CondFormats_SiStripObjects_DaqScopeModeAnalysis_H



