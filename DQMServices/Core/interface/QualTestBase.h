#ifndef _QUAL_TEST_BASE_H
#define _QUAL_TEST_BASE_H

#include <vector>
#include <string>
#include <TH1F.h>

#include "DQMServices/Core/interface/DQMDefinitions.h"

/// class T must be one of the usual histogram/profile objects: THX
/// for method notEnoughStats to be used...
template<class T>
class SimpleTest
{
 public:

  SimpleTest(void){min_entries_ = 0; badChannels_.clear();}
  virtual ~SimpleTest(void){}
  /// run the test (result: [0, 1] or <0 for failure)
  virtual float runTest(const T * const h) = 0;
 
  /// get vector of channels that failed test
  /// (not relevant for all quality tests!)
  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return badChannels_;}

 protected:
  /// set minimum # of entries needed
  void setMinimumEntries(unsigned N){min_entries_ = N;}
  /// true if histogram does not have enough statistics
  bool notEnoughStats(const T * const h) const
  {  
    if (h) return h->GetEntries() < min_entries_;
    return false;
  }
  /// minimum # of entries needed
  unsigned min_entries_;
  /// probability limits for warnings, errors
  float warningProb_, errorProb_;

  std::vector<dqm::me_util::Channel> badChannels_;

};

/// class T must be one of the usual histogram/profile objects: THX
template<class T>
class Comp2RefBase : public SimpleTest<T>
{
 public:
  
  Comp2RefBase(void) : SimpleTest<T>() {ref_ = 0;}
  virtual ~Comp2RefBase(void){}
  
  /// set reference object
  void setReference(const T * const h){ref_ = h;}
  /// true if reference object is null
  bool hasNullReference(void) const {return ref_ == 0;}
  /// run the test (result: [0, 1] or <0 for failure)
  virtual float runTest(const T * const h) = 0;
  
 protected:
  /// reference object
  const T * ref_;

};

#endif
