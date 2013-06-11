#include "RecoLocalCalo/HcalRecAlgos/src/HcalTDCReco.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"

HcalTDCReco::HcalTDCReco() 
{}

void HcalTDCReco::reconstruct(const HcalUpgradeDataFrame & digi, 
                              HBHERecHit & recHit) const
{
  int n = digi.size();
  double risingTime = -999.;
  double fallingTime = -999.;
  int signalBX = 5;              // NB: HARDWIRED !!!
  int nbins = 50; // as in HcalTDCParameters.h (SimCalorimetry/HcalSimAlgos)
  // start at bunch crossing 3 by default
  for(int i = 3; i < n; ++i)
  {
    unsigned tdc = digi.tdc(i);

    /* 
    unsigned rising = tdc & 0x7F;
    unsigned falling = (tdc >> 7) & 0x7F;
    */
    // temporary "unpacking" instead, which directly corresponds to 
    // SimCalorimetry/HcalSimAlgos/src/HcalTDC.cc  (nibs = 50...)
    // packedTDC = TDC_RisingEdge + (tdcBins*2) * TDC_FallingEdge;
    unsigned rising  =  tdc%100; 
    unsigned falling =  tdc/100;

    // only set the first time, avoiding "special" codes 
    if(risingTime < -998. && rising != 62 && rising != 63) {
      risingTime = rising*25./nbins + (i-signalBX)*25.;
    }
    if(fallingTime < -998. && falling != 62 && falling != 63) {
      fallingTime = falling*25./nbins + (i-signalBX)*25.;
    }    

    /*
    std::cout << " digi.tdc[" << i << "] = " << tdc 
	      << "  rising = " << rising << "  falling = " << falling
	      << "   Rt = " << risingTime
	      << "   Ft = " << fallingTime
	      << std::endl;
    */
  }
  recHit = HBHERecHit(recHit.id(), recHit.energy(), risingTime, fallingTime);
}
