#include <algorithm>

#include "RecoLocalCalo/HcalRecAlgos/interface/HFPreRecAlgo.h"

#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"

#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

HFQIE10Info HFPreRecAlgo::reconstruct(const QIE10DataFrame& digi,
                                      const int tsToUse,
                                      const HcalCoder& coder,
                                      const HcalCalibrations& calib) const
{
    // Conversion from TDC to ns for the QIE10 chip
    static const float qie10_tdc_to_ns = 0.5f;

    // TDC values produced in case the pulse is always above/below
    // the discriminator
    static const int qie10_tdc_code_overshoot = 62;
    static const int qie10_tdc_code_undershoot = 63;

    // Scrap the trailing edge time for now -- until the front-end
    // FPGA firmware is finalized and the description of the FPGA
    // output becomes available
    static const float timeFalling = HcalSpecialTimes::UNKNOWN_T_NOTDC;

    HFQIE10Info result;

    CaloSamples cs;
    coder.adc2fC(digi, cs);
    const int nRead = cs.size();

    // Number of raw samples to store in HFQIE10Info
    const int nStore = std::min(nRead, static_cast<int>(HFQIE10Info::N_RAW_MAX));

    if (sumAllTS_)
    {
        // This branch is intended for use with cosmic runs
        double charge = 0.0, energy = 0.0;
        HFQIE10Info::raw_type raw[HFQIE10Info::N_RAW_MAX];

        for (int ts=0; ts<nRead; ++ts)
        {
            const QIE10DataFrame::Sample s(digi[ts]);
            const int capid = s.capid();
            const float q = cs[ts] - calib.pedestal(capid);
            charge += q;
            energy += q*calib.respcorrgain(capid);
            if (ts < nStore)
                raw[ts] = s.wideRaw();
        }

        // Timing measurement does not appear to be useful here
        const float timeRising = HcalSpecialTimes::UNKNOWN_T_NOTDC;

        // The following HFQIE10Info arguments correspond to SOI
        // not stored in the raw data. Essentially, only charge
        // and energy are meaningful.
        result = HFQIE10Info(digi.id(), charge, energy,
                             timeRising, timeFalling,
                             raw, nStore, nStore);
    }
    else if (0 <= tsToUse && tsToUse < nRead)
    {
        const QIE10DataFrame::Sample s(digi[tsToUse]);
        const int capid = s.capid();
        const float charge = cs[tsToUse] - calib.pedestal(capid);
        const float energy = charge*calib.respcorrgain(capid);

        const int tdc = s.le_tdc();
        float timeRising = qie10_tdc_to_ns*tdc;
        if (tdc == qie10_tdc_code_overshoot)
            timeRising = HcalSpecialTimes::UNKNOWN_T_OVERSHOOT;
        else if (tdc == qie10_tdc_code_undershoot)
            timeRising = HcalSpecialTimes::UNKNOWN_T_UNDERSHOOT;

        // Figure out the window in the raw data
        // that we want to store. This window will
        // have the width given by "nStore" and
        // will start at "shift".
        int shift = 0;
        if (nRead > static_cast<int>(HFQIE10Info::N_RAW_MAX))
        {
            // Try to center the window on "tsToUse"
            const int winCenter = nStore/2;
            if (tsToUse > winCenter)
                shift = tsToUse - winCenter;
            if (shift + nStore > nRead)
                shift = nRead - nStore;
        }

        // Fill an array of raw values
        HFQIE10Info::raw_type raw[HFQIE10Info::N_RAW_MAX];
        for (int i=0; i<nStore; ++i)
            raw[i] = digi[i + shift].wideRaw();

        result = HFQIE10Info(digi.id(), charge, energy,
                             timeRising, timeFalling,
                             raw, nStore, tsToUse - shift);
    }
    return result;
}
