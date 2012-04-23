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
  for(int i=0; i < n; ++i)
  {
    unsigned tdc = digi.tdc(i);
    unsigned rising = tdc & 0x1F;
    unsigned falling = (tdc >> 5) & 0x1F;
    // only set the first time
    if(risingTime < -998. && rising != 0 && rising != 31) {
      risingTime = rising*25./32. + (i-signalBX)*25.;
    }
    if(fallingTime < -998. && falling != 0 && falling != 31) {
      fallingTime = falling*25./32. + (i-signalBX)*25.;
    }    
  }
  recHit = HcalUpgradeRecHit(recHit.id(), recHit.energy(), risingTime, fallingTime);
}
