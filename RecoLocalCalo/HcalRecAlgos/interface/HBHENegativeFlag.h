#ifndef HBHENegativeFlag_H
#define HBHENegativeFlag_H

//---------------------------------------------------------------------------
// Negative filter algorithms for HBHE noise flagging
//---------------------------------------------------------------------------

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CondFormats/HcalObjects/interface/HBHENegativeEFilter.h"

class HBHENegativeFlagSetter
{
public:
    inline HBHENegativeFlagSetter() : filter_(0) {}

    inline void configFilter(const HBHENegativeEFilter* f) {filter_ = f;}

    void setPulseShapeFlags(HBHERecHit& hbhe, const HBHEDataFrame &digi,
                            const HcalCoder &coder, const HcalCalibrations &calib);
private:
    const HBHENegativeEFilter* filter_;
};

#endif
