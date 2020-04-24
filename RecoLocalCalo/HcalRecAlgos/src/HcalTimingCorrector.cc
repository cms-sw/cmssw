#include "../interface/HcalTimingCorrector.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

// Using defined flags instead of hardcoding bits 2 and 3

namespace HcalTimingCorrector_impl {
  template <class T, class V>
  void makeCorrection(T &rechit, const V &digi, int favorite_capid) {
    if (digi.fiberIdleOffset() == -1000) return; //reserved for bad stuff

    bool bad = false;
    //convention, -1 means digi is early, need to add
    //            +1 means digi is late, need to subtract
    int capid_shift = (favorite_capid - digi[0].capid());
    capid_shift %= 4;
    while (capid_shift < 0) capid_shift += 4;
    
    int idle_shift = digi.fiberIdleOffset();
    idle_shift %= 4;
    while (idle_shift < 0) idle_shift += 4;
    
    if (capid_shift != idle_shift) {
      bad = true;
    }
    else if (capid_shift == 0) {
      // do nothing, all is well
    }
    else if (capid_shift == 2) {
      bad = true;
    }
    else {
      /*
      std::cout << "capid_shift " << capid_shift 
		<< " fiberOffset " << digi.fiberIdleOffset() << std::endl;
      */
      rechit = T(rechit.id(), rechit.energy(), rechit.time() +
		 ( capid_shift == 1 ? -25 : +25 )
		 );
      rechit.setFlagField(1,(capid_shift == 1 ? 
			     HcalCaloFlagLabels::TimingSubtractedBit : 
			     HcalCaloFlagLabels::TimingAddedBit ));
    }

    if (bad) {
      /*
	std::cout << "BAD capid_shift " << capid_shift 
	<< " fiberOffset " << digi.fiberIdleOffset() << std::endl;
      */      
      rechit.setFlagField(1,HcalCaloFlagLabels::TimingErrorBit);
    }
  }
}

using namespace HcalTimingCorrector_impl;

HcalTimingCorrector::HcalTimingCorrector()
{}

HcalTimingCorrector::~HcalTimingCorrector()
{}

void HcalTimingCorrector::Correct(HBHERecHit& rechit, const HBHEDataFrame& digi, int favorite_capid)
{
  makeCorrection<HBHERecHit, HBHEDataFrame>(rechit, digi, favorite_capid);
}

void HcalTimingCorrector::Correct(HORecHit& rechit, const HODataFrame& digi, int favorite_capid)
{
  makeCorrection<HORecHit, HODataFrame>(rechit, digi, favorite_capid);
}

void HcalTimingCorrector::Correct(HFRecHit& rechit, const HFDataFrame& digi, int favorite_capid)
{
  makeCorrection<HFRecHit, HFDataFrame>(rechit, digi, favorite_capid);
}

