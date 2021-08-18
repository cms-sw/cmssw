#ifndef DataFormats_L1Trigger_MuonShower_h
#define DataFormats_L1Trigger_MuonShower_h

/*
  This class is derived from the L1Candidate primarily to interface easily
  with the Global Muon Trigger. In the trigger system the MuonShower object
  carries only up to 4 bits of information, 2 for in-time showers,
  2 for out-of-time showers.
*/

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"

namespace l1t {

  class MuonShower;
  typedef BXVector<MuonShower> MuonShowerBxCollection;
  typedef edm::Ref<MuonShowerBxCollection> MuonShowerRef;
  typedef edm::RefVector<MuonShowerBxCollection> MuonShowerRefVector;
  typedef std::vector<MuonShowerRef> MuonShowerVectorRef;

  typedef ObjectRefBxCollection<MuonShower> MuonShowerRefBxCollection;
  typedef ObjectRefPair<MuonShower> MuonShowerRefPair;
  typedef ObjectRefPairBxCollection<MuonShower> MuonShowerRefPairBxCollection;

  class MuonShower : public L1Candidate {
  public:
    MuonShower(bool oneNominalInTime = false,
               bool oneNominalOutOfTime = false,
               bool twoLooseInTime = false,
               bool twoLooseOutOfTime = false);

    ~MuonShower() override;

    void setOneNominalInTime(const bool bit) { isOneNominalInTime_ = bit; }
    void setOneNominalOutOfTime(const bool bit) { isOneNominalOutOfTime_ = bit; }
    void setTwoLooseInTime(const bool bit) { isTwoLooseInTime_ = bit; }
    void setTwoLooseOutOfTime(const bool bit) { isTwoLooseOutOfTime_ = bit; }

    bool isValid() const;
    bool isOneNominalInTime() const { return isOneNominalInTime_; }
    bool isOneNominalOutOfTime() const { return isOneNominalOutOfTime_; }
    bool isTwoLooseInTime() const { return isTwoLooseInTime_; }
    bool isTwoLooseOutOfTime() const { return isTwoLooseOutOfTime_; }

    virtual bool operator==(const l1t::MuonShower& rhs) const;
    virtual inline bool operator!=(const l1t::MuonShower& rhs) const { return !(operator==(rhs)); };

  private:
    // Run-3 definitions as provided in DN-20-033
    // in time and out-of-time qualities. only 2 bits each.
    bool isOneNominalInTime_;
    bool isOneNominalOutOfTime_;
    bool isTwoLooseInTime_;
    bool isTwoLooseOutOfTime_;
  };

}  // namespace l1t

#endif
