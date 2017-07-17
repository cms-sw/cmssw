#ifndef RecoLocalCalo_HcalRecAlgos_HFPreRecAlgo_h_
#define RecoLocalCalo_HcalRecAlgos_HFPreRecAlgo_h_

#include "DataFormats/HcalRecHit/interface/HFQIE10Info.h"

class QIE10DataFrame;
class HcalCoder;
class HcalCalibrations;

class HFPreRecAlgo
{
public:
    inline explicit HFPreRecAlgo(const bool sumAllTS) : sumAllTS_(sumAllTS) {}

    inline ~HFPreRecAlgo() {}

    HFQIE10Info reconstruct(const QIE10DataFrame& digi,
                            int tsToUse,
                            const HcalCoder& coder,
                            const HcalCalibrations& calibs) const;
private:
    bool sumAllTS_;
};

#endif // RecoLocalCalo_HcalRecAlgos_HFPreRecAlgo_h_
