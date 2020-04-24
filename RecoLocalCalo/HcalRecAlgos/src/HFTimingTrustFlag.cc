#include "RecoLocalCalo/HcalRecAlgos/interface/HFTimingTrustFlag.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

namespace HFTimingTrust 
{
  // Template class checks HF timing, sets rechit 
  // timing status bits according to its uncertainty;

  double timerr_hf(double rectime, double ampl);

  template <class T, class V>
  void checkHFTimErr(T &rechit, const V &digi, int level1, int level2) 
  {
    // Get rechit timing
    double rectime = rechit.time();

    if (rectime>-100 && rectime<250) {

      // Get signal from digi
      double ampl=0; double sum=0; int maxI = -1;
      for (int i=0;i<digi.size();++i) {
	sum += digi.sample(i).nominal_fC();
	if (digi.sample(i).nominal_fC()>ampl) {
	  ampl = digi.sample(i).nominal_fC();
	  maxI = i;
	}
      }
      if (ampl>1 && maxI>0 && maxI<digi.size()-1) {
	ampl = ampl + digi.sample(maxI-1).nominal_fC() + digi.sample(maxI+1).nominal_fC();
	ampl -= (sum-ampl)*3.0/(digi.size()-3);
	if (ampl>3) {
	  int timerr = (int) timerr_hf(rectime,ampl);
	  uint32_t status=0;
	  if (timerr<0) status = 3; // unreconstructable time
	  else if (timerr<=level1) status = 0; // time reconstructed; precision better than level1 value
	  else if (timerr<=level2) status = 1; // precision worse than level1 value
	  else status = 2; //precision worse than level 2 value
	  rechit.setFlagField(status,HcalCaloFlagLabels::HFTimingTrustBits,2);
	  return;
	}
      }

    } // if (rectime > -100 && rectime < -250)
    // if rectime outside expected range, set flag field to 3 (unreconstructable)?
    rechit.setFlagField(3,HcalCaloFlagLabels::HFTimingTrustBits,2);
    return;
  }

  static const float hfterrlut[195] = { 
    3.42,  3.04,  9.18,  8.97,  8.49,  8.08,  8.01,  8.30,  8.75,  8.22,  7.21,  5.04,  2.98,
    2.04,  1.22,  7.58,  7.79,  7.11,  6.93,  7.03,  7.27,  7.23,  6.53,  4.59,  2.40,  1.46,
    1.31,  0.42,  5.95,  6.48,  6.29,  5.84,  5.97,  6.31,  6.00,  4.37,  2.37,  1.03,  0.72,
    0.81,  0.27,  3.98,  5.57,  5.04,  5.10,  5.21,  5.18,  4.22,  2.23,  1.07,  0.66,  0.40,
    0.48,  0.17,  2.51,  4.70,  4.28,  4.29,  4.36,  3.84,  2.40,  1.15,  0.68,  0.40,  0.24,
    0.29,  0.11,  0.81,  3.71,  3.47,  3.48,  3.52,  2.58,  1.25,  0.71,  0.41,  0.26,  0.16,
    0.16,  0.08,  0.27,  2.88,  2.63,  2.76,  2.33,  1.31,  0.72,  0.44,  0.27,  0.16,  0.11,
    0.10,  0.06,  0.15,  2.11,  2.00,  1.84,  1.46,  0.79,  0.45,  0.26,  0.17,  0.10,  0.08,
    0.05,  0.04,  0.10,  1.58,  1.49,  1.25,  0.90,  0.48,  0.29,  0.17,  0.10,  0.06,  0.06,
    0.02,  0.03,  0.06,  1.26,  1.03,  0.77,  0.57,  0.30,  0.18,  0.11,  0.06,  0.04,  0.05,
    0.01,  0.02,  0.04,  0.98,  0.66,  0.47,  0.39,  0.18,  0.11,  0.07,  0.04,  0.03,  0.04,
    0.01,  0.02,  0.02,  0.86,  0.44,  0.30,  0.27,  0.11,  0.07,  0.04,  0.03,  0.02,  0.04,
    0.01,  0.02,  0.02,  0.80,  0.30,  0.21,  0.17,  0.07,  0.04,  0.03,  0.02,  0.01,  0.04,
    0.01,  0.02,  0.01,  0.76,  0.22,  0.17,  0.12,  0.05,  0.03,  0.02,  0.01,  0.01,  0.04,
    0.01,  0.02,  0.01,  0.76,  0.17,  0.14,  0.09,  0.03,  0.02,  0.01,  0.01,  0.01,  0.04
  };

  double timerr_hf(double rectime, double ampl)
  {
    int itim,iampl,index;
    double tim;
    if (rectime>0) tim=rectime-((int)(rectime/25))*25;
    else tim=rectime-((int)(rectime/25))*25+25;
    itim = (int) (tim/2.0);

    iampl=0;
    static const double bampl[15]={3,5,8,12,19,30,47,73,115,182,287,452,712,1120,1766};
    if (ampl>=bampl[14]) iampl=14;
    else {
      for (int i=1;i<=14;i++) {
	if (ampl<bampl[i]) {
	  iampl=i-1;
	  break;
	}
      }
    }

    index = itim + iampl*13;

    double y1 = hfterrlut[index];
    double y2 = 0;
    double v1 = y1;
    if (itim<12) {
      y2 = hfterrlut[index+1];
      v1 = y1 + (y2-y1)*(tim/2.0-(float)itim);
    }
    double yy1 = 0;
    double yy2 = 0;
    double v2 = 0;
    if (iampl<14) {
      yy1 = hfterrlut[index+13];
      if (itim==12) v2 = yy1;
      else {
	yy2 = hfterrlut[index+14];
	v2 = yy1 + (yy2-yy1)*(tim/2.0-(float)itim);
      }
      v1 = v1 + (v2-v1)*(ampl-bampl[iampl])/(bampl[iampl+1]-bampl[iampl]);
    }

    return v1;
  }

}

using namespace HFTimingTrust;

HFTimingTrustFlag::HFTimingTrustFlag()
{
  HFTimingTrustLevel1_ = 1; // time precision 1ns
  HFTimingTrustLevel2_ = 4; // time precision 4ns
}

HFTimingTrustFlag::HFTimingTrustFlag(int level1, int level2)
{
  HFTimingTrustLevel1_ = level1; // allow user to set t-trust level
  HFTimingTrustLevel2_ = level2; 
}

HFTimingTrustFlag::~HFTimingTrustFlag()
{}

void HFTimingTrustFlag::setHFTimingTrustFlag(HFRecHit& rechit, const HFDataFrame& digi)
{
  checkHFTimErr<HFRecHit, HFDataFrame>(rechit, digi, HFTimingTrustLevel1_, HFTimingTrustLevel2_);
  return;
}

