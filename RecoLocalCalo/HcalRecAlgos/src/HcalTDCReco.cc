#include "RecoLocalCalo/HcalRecAlgos/src/HcalTDCReco.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalUpgradeRecHit.h"

HcalTDCReco::HcalTDCReco() 
{}

void HcalTDCReco::reconstruct(const HcalUpgradeDataFrame & digi, 
                              HcalUpgradeRecHit & recHit) const
{
  int n = digi.size();
  double risingTime = -999.;
  double fallingTime = -999.;
  int signalBX = 5;
  // start at bunch crossing 3
  for(int i=3; i < n; ++i)
  {
    unsigned tdc = digi.tdc(i);
    unsigned rising = tdc & 0x7F;
    // altered packing to tdcBins*2=7
    unsigned falling = (tdc >> 7) & 0x7F;
    // only set the first time
    if(risingTime < -998. && rising != 64 && rising != 65) {
      risingTime = rising*25./64. + (i-signalBX)*25.;
    }
    if(fallingTime < -998. && falling != 64 && falling != 65) {
      fallingTime = falling*25./64. + (i-signalBX)*25.;
    }    
  }
  recHit = HcalUpgradeRecHit(recHit.id(), recHit.energy(), risingTime, fallingTime);
}
