#ifndef DataFormats_L1Trigger_RegionalMuonShower_h
#define DataFormats_L1Trigger_RegionalMuonShower_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"

#include "RegionalMuonCandFwd.h"  // For tftype.

namespace l1t {

  class RegionalMuonShower;
  typedef BXVector<RegionalMuonShower> RegionalMuonShowerBxCollection;
  typedef ObjectRefBxCollection<RegionalMuonShower> RegionalMuonShowerRefBxCollection;
  typedef ObjectRefPair<RegionalMuonShower> RegionalMuonShowerRefPair;
  typedef ObjectRefPairBxCollection<RegionalMuonShower> RegionalMuonShowerRefPairBxCollection;

  class RegionalMuonShower {
  public:
    RegionalMuonShower(bool oneNominalInTime = false,
                       bool oneNominalOutOfTime = false,
                       bool twoLooseInTime = false,
                       bool twoLooseOutOfTime = false,
                       bool oneLooseInTime = false,
                       bool oneTightInTime = false,
                       bool oneTightOutOfTime = false);

    ~RegionalMuonShower();

    void setOneNominalInTime(const bool bit) { isOneNominalInTime_ = bit; }
    void setOneNominalOutOfTime(const bool bit) { isOneNominalOutOfTime_ = bit; }
    void setOneTightInTime(const bool bit) { isOneTightInTime_ = bit; }
    void setOneTightOutOfTime(const bool bit) { isOneTightOutOfTime_ = bit; }
    void setTwoLooseOutOfTime(const bool bit) { isTwoLooseOutOfTime_ = bit; }
    void setTwoLooseInTime(const bool bit) { isTwoLooseInTime_ = bit; }
    void setOneLooseInTime(const bool bit) { isOneLooseInTime_ = bit; }

    /// Set the processor ID, track-finder type. From these two, the link is set
    void setTFIdentifiers(int processor, tftype trackFinder);

    bool isValid() const;
    bool isOneNominalInTime() const { return isOneNominalInTime_; }
    bool isOneNominalOutOfTime() const { return isOneNominalOutOfTime_; }
    bool isOneTightInTime() const { return isOneTightInTime_; }
    bool isOneTightOutOfTime() const { return isOneTightOutOfTime_; }
    bool isTwoLooseInTime() const { return isTwoLooseInTime_; }
    bool isTwoLooseOutOfTime() const { return isTwoLooseOutOfTime_; }
    bool isOneLooseInTime() const { return isOneLooseInTime_; }

    /// Get link on which the MicroGMT receives the candidate
    const int link() const { return link_; };
    /// Get processor ID on which the candidate was found (0..5 for OMTF/EMTF; 0..11 for BMTF)
    const int processor() const { return processor_; };
    /// Get track-finder which found the muon (bmtf, emtf_pos/emtf_neg or omtf_pos/omtf_neg)
    const tftype trackFinderType() const { return trackFinder_; };

    bool operator==(const l1t::RegionalMuonShower& rhs) const;
    inline bool operator!=(const l1t::RegionalMuonShower& rhs) const { return !(operator==(rhs)); };

  private:
    // Run-3 definitions as provided in DN-20-033
    // in time and out-of-time qualities. only 2 bits each.
    bool isOneNominalInTime_;
    bool isOneNominalOutOfTime_;
    bool isOneTightInTime_;
    bool isOneTightOutOfTime_;
    bool isTwoLooseInTime_;
    bool isTwoLooseOutOfTime_;
    bool isOneLooseInTime_;
    int link_;
    int processor_;
    tftype trackFinder_;
  };

}  // namespace l1t

#endif
