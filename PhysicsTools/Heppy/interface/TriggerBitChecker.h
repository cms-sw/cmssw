#ifndef PhysicsTools_Heppy_TriggerBitChecker_h
#define PhysicsTools_Heppy_TriggerBitChecker_h

#include <vector>
#include <string>
#include <iostream>
#include <cassert>
#include <type_traits>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "FWCore/Common/interface/EventBase.h"

namespace heppy {

  class TriggerBitChecker {
  public:
    struct pathStruct {
      pathStruct(const std::string &s) : pathName(s), first(0), last(99999999) {}
      pathStruct() : pathName(), first(0), last(99999999) {}
      std::string pathName;
      unsigned int first;
      unsigned int last;
    };

    TriggerBitChecker(const std::string &path = "DUMMY");
    TriggerBitChecker(const std::vector<std::string> &paths);
    ~TriggerBitChecker() {}

    bool check(const edm::EventBase &event, const edm::TriggerResults &result) const;

    bool check_unprescaled(const edm::EventBase &event,
                           const edm::TriggerResults &result_tr,
                           const pat::PackedTriggerPrescales &result) const;

    // method templated to force correct choice of output type
    // (as part of deprecating integer types for trigger prescales)
    template <typename T = int>
    T getprescale(const edm::EventBase &event,
                  const edm::TriggerResults &result_tr,
                  const pat::PackedTriggerPrescales &result) const;

  private:
    // list of path name prefixes
    std::vector<pathStruct> paths_;

    mutable edm::ParameterSetID lastID_;
    mutable std::vector<unsigned int> indices_;

    /// sync indices with path names
    void syncIndices(const edm::EventBase &event, const edm::TriggerResults &result) const;
    pathStruct returnPathStruct(const std::string &path) const;

    /// executes a 'rm -rf *' in current directory
    void rmstar();
  };

  template <typename T>
  T TriggerBitChecker::getprescale(const edm::EventBase &event,
                                   const edm::TriggerResults &result_tr,
                                   const pat::PackedTriggerPrescales &result) const {
    static_assert(std::is_same_v<T, double>,
                  "\n\n\tPlease use getprescale<double> "
                  "(other types for trigger prescales are not supported anymore by TriggerBitChecker)");
    if (result_tr.parameterSetID() != lastID_) {
      syncIndices(event, result_tr);
      lastID_ = result_tr.parameterSetID();
    }
    if (indices_.empty()) {
      return -999;
    }
    if (indices_.size() > 1) {
      std::cout << " trying to get prescale for multiple trigger objects at the same time" << std::endl;
      assert(0);
    }

    return result.getPrescaleForIndex<T>(*(indices_.begin()));
  }

}  // namespace heppy

#endif
