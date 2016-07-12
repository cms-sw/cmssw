#ifndef RecoLocalCalo_HcalRecAlgos_HFPreRecAlgo_h_
#define RecoLocalCalo_HcalRecAlgos_HFPreRecAlgo_h_

#include "DataFormats/HcalRecHit/interface/HFQIE10Info.h"

class QIE10DataFrame;
class HcalCoder;
class HcalCalibrations;

class HFPreRecAlgo
{
public:
    inline HFPreRecAlgo() {}
    inline ~HFPreRecAlgo() {}

    HFQIE10Info reconstruct(const QIE10DataFrame& digi,
                            int tsToUse,
                            const HcalCoder& coder,
                            const HcalCalibrations& calibs);
};

#endif // RecoLocalCalo_HcalRecAlgos_HFPreRecAlgo_h_
