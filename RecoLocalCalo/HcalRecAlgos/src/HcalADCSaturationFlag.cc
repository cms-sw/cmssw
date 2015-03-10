#include "../interface/HcalADCSaturationFlag.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"



namespace HcalSaturation 
{
  // Template class that loops over digi collection, sets rechit 
  // saturation status bit on if ADC count is >= SaturationLevel_;

  template <class T, class V>
  void checkADCSaturation(T &rechit, const V &digi, int level) 
  {
    // Loop over digi
    for (int i=0;i<digi.size();++i)
      {
	if (digi.sample(i).adc()>=level)
	  {
	    rechit.setFlagField(1,HcalCaloFlagLabels::ADCSaturationBit);
	    break;
	  }
      }
    return;
  }
}

using namespace HcalSaturation;

HcalADCSaturationFlag::HcalADCSaturationFlag()
{
  SaturationLevel_=127; // default saturation level (7-bit QIE)
}

HcalADCSaturationFlag::HcalADCSaturationFlag(int level)
{
  SaturationLevel_=level; // allow user to specify saturation level
}

HcalADCSaturationFlag::~HcalADCSaturationFlag()
{}

void HcalADCSaturationFlag::setSaturationFlag(HBHERecHit& rechit, const HBHEDataFrame& digi)
{
  checkADCSaturation<HBHERecHit, HBHEDataFrame>(rechit, digi, SaturationLevel_);
  return;
}

void HcalADCSaturationFlag::setSaturationFlag(HORecHit& rechit, const HODataFrame& digi)
{
  checkADCSaturation<HORecHit, HODataFrame>(rechit, digi, SaturationLevel_);
  return;
}

void HcalADCSaturationFlag::setSaturationFlag(HFRecHit& rechit, const HFDataFrame& digi)
{
  checkADCSaturation<HFRecHit, HFDataFrame>(rechit, digi, SaturationLevel_);
  return;
}

void HcalADCSaturationFlag::setSaturationFlag(ZDCRecHit& rechit, const ZDCDataFrame& digi)
{
  checkADCSaturation<ZDCRecHit, ZDCDataFrame>(rechit, digi, SaturationLevel_);
  return;
}
