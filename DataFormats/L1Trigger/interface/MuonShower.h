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

    // This makes sense as long as the one-nominal shower case is mapped
    // onto 0b01 and the two-loose shower case is mapped onto 0b10.
    // A third (unused option) is 0b11
    void setOneNominalInTime(const bool bit) {
      isOneNominalInTime_ = bit;
      mus0_ = bit;
    }
    void setOneNominalOutOfTime(const bool bit) {
      isOneNominalOutOfTime_ = bit;
      musOutOfTime0_ = bit;
    }
    void setTwoLooseInTime(const bool bit) {
      isTwoLooseInTime_ = bit;
      mus1_ = bit;
    }
    void setTwoLooseOutOfTime(const bool bit) {
      isTwoLooseOutOfTime_ = bit;
      musOutOfTime1_ = bit;
    }

    void setMus0(const bool bit) { mus0_ = bit; }
    void setMus1(const bool bit) { mus1_ = bit; }
    void setMusOutOfTime0(const bool bit) { musOutOfTime0_ = bit; }
    void setMusOutOfTime1(const bool bit) { musOutOfTime1_ = bit; }

    bool isValid() const;
    bool isOneNominalInTime() const { return isOneNominalInTime_; }
    bool isOneNominalOutOfTime() const { return isOneNominalOutOfTime_; }
    bool isTwoLooseInTime() const { return isTwoLooseInTime_; }
    bool isTwoLooseOutOfTime() const { return isTwoLooseOutOfTime_; }

    bool mus0() const { return mus0_; }
    bool mus1() const { return mus1_; }
    bool musOutOfTime0() const { return musOutOfTime0_; }
    bool musOutOfTime1() const { return musOutOfTime1_; }

    virtual bool operator==(const l1t::MuonShower& rhs) const;
    virtual inline bool operator!=(const l1t::MuonShower& rhs) const { return !(operator==(rhs)); };

  private:
    // Run-3 definitions as provided in DN-20-033
    // in time and out-of-time qualities. only 2 bits each.
    bool isOneNominalInTime_;
    bool isOneNominalOutOfTime_;
    bool isTwoLooseInTime_;
    bool isTwoLooseOutOfTime_;

    // The data members below represent the same data as the 4 members above.
    // They are needed in order to interface with the uGT GlobalBoard class.
    // 2 bits for the in-time showers
    bool mus0_;
    bool mus1_;
    // 2 bits for the out-of-time showers
    bool musOutOfTime0_;
    bool musOutOfTime1_;
  };

}  // namespace l1t

#endif
