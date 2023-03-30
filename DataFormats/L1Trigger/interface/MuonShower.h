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
               bool twoLooseOutOfTime = false,
               bool oneTightInTime = false,
               bool oneTightOutOfTime = false,
               bool twoLooseDiffSectorsIntime = false);

    ~MuonShower() override;

    /*
     * We currently have three valid cases:
     * - 1 nominal shower (baseline trigger for physics at Run-3)
     * - 1 tight shower (backup trigger)
     * - 2 loose showers in different sectors (to extend the physics reach)
     *
     * uGT receives shower information on five bits:
     * - 2 bits for in-time showers (mus0, mus1 in UTM)
     * - 2 bits for out-of-time showers (musOutOfTime0, musOutOfTime1 in UTM)
     * - 1 bit for the newly added two loose showers in different sectors (mus2 in UTM)
     * Disclaimer: The notion of "in time" and "out of time" showers is outdated, as there
     * are no plans for out of time showers anymore, however the naming has been kept here
     * to avoid confusion with the way things have been named from the start.
     *
     * The mapping as it is currently understood is:
     * - 1 nominal shower -> mus0
     * - 1 tight shower -> mus1
     * - 2 loose showers in differnt sectors -> mus2
     */

    void setOneNominalInTime(const bool bit) { oneNominalInTime_ = bit; }
    void setOneTightInTime(const bool bit) { oneTightInTime_ = bit; }
    void setTwoLooseDiffSectorsInTime(const bool bit) { twoLooseDiffSectorsInTime_ = bit; }
    void setMusOutOfTime0(const bool bit) { musOutOfTime0_ = bit; }
    void setMusOutOfTime1(const bool bit) { musOutOfTime1_ = bit; }

    bool isOneNominalInTime() const { return oneNominalInTime_; }
    bool isOneTightInTime() const { return oneTightInTime_; }
    bool isTwoLooseDiffSectorsInTime() const { return twoLooseDiffSectorsInTime_; }
    bool musOutOfTime0() const { return musOutOfTime0_; }
    bool musOutOfTime1() const { return musOutOfTime1_; }

    // at least one bit must be valid
    bool isValid() const;

    // to be developed during Run-3
    bool isTwoLooseInTime() const { return false; }
    // these options require more study
    bool isOneNominalOutOfTime() const { return false; }
    bool isTwoLooseOutOfTime() const { return false; }
    bool isOneTightOutOfTime() const { return false; }

    virtual bool operator==(const l1t::MuonShower& rhs) const;
    virtual inline bool operator!=(const l1t::MuonShower& rhs) const { return !(operator==(rhs)); };

  private:
    // Run-3 definitions as provided in DN-20-033
    // in time and out-of-time qualities. only 2 bits each.
    bool oneNominalInTime_;
    bool oneTightInTime_;
    bool twoLooseDiffSectorsInTime_;
    bool musOutOfTime0_;
    bool musOutOfTime1_;
  };

}  // namespace l1t

#endif
