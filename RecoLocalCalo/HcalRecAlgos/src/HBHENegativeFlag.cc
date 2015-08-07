#include "RecoLocalCalo/HcalRecAlgos/interface/HBHENegativeFlag.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

void HBHENegativeFlagSetter::setPulseShapeFlags(
    HBHERecHit &hbhe, 
    const HBHEDataFrame &digi,
    const HcalCoder &coder, 
    const HcalCalibrations &calib)
{
   if (filter_)
   {
      CaloSamples cs;
      coder.adc2fC(digi,cs);
      const int nRead = cs.size();

      double ts[CaloSamples::MAXSAMPLES];
      for (int i=0; i < nRead; i++)
      {
         const int capid = digi[i].capid();
         ts[i] = cs[i] - calib.pedestal(capid);
      }

      const bool passes = filter_->checkPassFilter(hbhe.id(), &ts[0], nRead);
      if (!passes)
          hbhe.setFlagField(1, HcalCaloFlagLabels::HBHENegativeNoise);
   }
}
