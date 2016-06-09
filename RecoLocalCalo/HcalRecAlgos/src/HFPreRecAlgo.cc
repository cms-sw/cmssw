#include "RecoLocalCalo/HcalRecAlgos/interface/HFPreRecAlgo.h"

#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"

#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

HFQIE10Info HFPreRecAlgo::reconstruct(const QIE10DataFrame& digi,
                                      const int tsToUse,
                                      const HcalCoder& coder,
                                      const HcalCalibrations& calib)
{
    HFQIE10Info result;

    CaloSamples cs;
    coder.adc2fC(digi, cs);
    const int nRead = cs.size();

    if (0 <= tsToUse && tsToUse < nRead)
    {
        const QIE10DataFrame::Sample s(digi[tsToUse]);
        const int capid = s.capid();
        const float charge = cs[tsToUse] - calib.pedestal(capid);
        const float energy = charge*calib.respcorrgain(capid);
        const float timeRising = s.le_tdc();
        const float timeFalling = s.te_tdc();

        // Set presample raw data. Map "tsToUse" index
        // to the last raw datum remembered. We will also
        // remember up to HFQIE10Info::N_RAW_MAX - 1
        // previous time slices.
        HFQIE10Info::raw_type raw[HFQIE10Info::N_RAW_MAX];
        int which = HFQIE10Info::N_RAW_MAX - 1;
        raw[which--] = s.wideRaw();
        for (int iSample = tsToUse-1; iSample >= 0 && which >= 0; --iSample)
        {
            const QIE10DataFrame::Sample samp(digi[iSample]);
            raw[which--] = samp.wideRaw();
        }
        ++which;

        result = HFQIE10Info(digi.id(), charge, energy,
                             timeRising, timeFalling, raw+which,
                             HFQIE10Info::N_RAW_MAX-which);
    }
    return result;
}
