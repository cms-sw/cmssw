#ifndef RecoLocalCalo_HcalRecAlgos_HFFlexibleTimeCheck_h_
#define RecoLocalCalo_HcalRecAlgos_HFFlexibleTimeCheck_h_

#include "RecoLocalCalo/HcalRecAlgos/interface/HFSimpleTimeCheck.h"
#include "CondFormats/HcalObjects/interface/HFPhase1PMTParams.h"

class HFFlexibleTimeCheck : public HFSimpleTimeCheck
{
public:
    using HFSimpleTimeCheck::HFSimpleTimeCheck;

    inline ~HFFlexibleTimeCheck() override {}

    // Unlike HFSimpleTimeCheck, this algorithm is configurable
    inline bool isConfigurable() const override {return true;}
    inline bool configure(const AbsHcalAlgoData* config) override
    {
        algoConf_ = dynamic_cast<const HFPhase1PMTParams*>(config);
        return algoConf_;
    }

    HFRecHit reconstruct(const HFPreRecHit& prehit,
                                 const HcalCalibrations& calibs,
                                 const bool flaggedBadInDB[2],
                                 bool expectSingleAnodePMT) override;
protected:
    unsigned determineAnodeStatus(unsigned anodeNumber,
                                          const HFQIE10Info& anode,
                                          bool* isTimingReliable) const override;
private:
    // Algorihm configuration data. We do not manage these pointers.
    const HFPhase1PMTParams* algoConf_ = nullptr;
    const HFPhase1PMTData* pmtInfo_ = nullptr;
};

#endif // RecoLocalCalo_HcalRecAlgos_HFFlexibleTimeCheck_h_
