#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"

//#define DEBUG_SiStripZeroSuppression_
//#define ML_DEBUG
using namespace std;

void SiStripFedZeroSuppression::init(const edm::EventSetup& es) {
  if (noiseWatcher_.check(es)) {
    noise_ = &es.getData(noiseToken_);
  }
  if (thresholdWatcher_.check(es)) {
    threshold_ = &es.getData(thresholdToken_);
  }
}

namespace {

  constexpr uint8_t zeroTh = 0;
  constexpr uint8_t lowTh = 1;
  constexpr uint8_t highTh = 2;

  struct Payload {
    uint8_t stat;
    uint8_t statPrev;
    uint8_t statNext;
    uint8_t statMaxNeigh;
    uint8_t statPrev2;
    uint8_t statNext2;
  };

  constexpr bool isDigiValid(Payload const& data, uint16_t FEDalgorithm) {
    // Decide if this strip should be accepted.
    bool accept = false;
    switch (FEDalgorithm) {
      case 1:
        accept = (data.stat >= lowTh);
        break;
      case 2:
        accept = (data.stat >= highTh || (data.stat >= lowTh && data.statMaxNeigh >= lowTh));
        break;
      case 3:
        accept = (data.stat >= highTh || (data.stat >= lowTh && data.statMaxNeigh >= highTh));
        break;
      case 4:
        accept = ((data.stat >= highTh)     //Test for adc>highThresh (same as algorithm 2)
                  || ((data.stat >= lowTh)  //Test for adc>lowThresh, with neighbour adc>lowThresh (same as algorithm 2)
                      && (data.statMaxNeigh >= lowTh)) ||
                  ((data.stat < lowTh)             //Test for adc<lowThresh
                   && (((data.statPrev >= highTh)  //with both neighbours>highThresh
                        && (data.statNext >= highTh)) ||
                       ((data.statPrev >= highTh)    //OR with previous neighbour>highThresh and
                        && (data.statNext >= lowTh)  //both the next neighbours>lowThresh
                        && (data.statNext2 >= lowTh)) ||
                       ((data.statNext >= highTh)    //OR with next neighbour>highThresh and
                        && (data.statPrev >= lowTh)  //both the previous neighbours>lowThresh
                        && (data.statPrev2 >= lowTh)) ||
                       ((data.statNext >= lowTh)      //OR with both next neighbours>lowThresh and
                        && (data.statNext2 >= lowTh)  //both the previous neighbours>lowThresh
                        && (data.statPrev >= lowTh) && (data.statPrev2 >= lowTh)))));
        break;
      case 5:
        accept = true;  // zero removed in conversion
        break;
    }
    return accept;
  }

}  // namespace

void SiStripFedZeroSuppression::suppress(const std::vector<SiStripDigi>& in,
                                         std::vector<SiStripDigi>& selectedSignal,
                                         uint32_t detID) {
  suppress(in, selectedSignal, detID, *noise_, *threshold_);
}

void SiStripFedZeroSuppression::suppress(const std::vector<SiStripDigi>& in,
                                         std::vector<SiStripDigi>& selectedSignal,
                                         uint32_t detID,
                                         const SiStripNoises& noise,
                                         const SiStripThreshold& threshold) {
  selectedSignal.clear();
  size_t inSize = in.size();
  if (inSize == 0) {
    return;
  }

  SiStripNoises::Range detNoiseRange = noise.getRange(detID);
  SiStripThreshold::Range detThRange = threshold.getRange(detID);

  // reserving more than needed, but quicker than one at a time
  selectedSignal.reserve(inSize);

  // load status
  uint8_t stat[inSize];
  for (size_t i = 0; i < inSize; i++) {
    auto strip = (uint32_t)in[i].strip();

    auto ladc = in[i].adc();
    assert(ladc > 0);

    auto thresholds = threshold.getData(strip, detThRange);

    auto highThresh = static_cast<int16_t>(thresholds.getHth() * noise.getNoiseFast(strip, detNoiseRange) + 0.5f);
    auto lowThresh = static_cast<int16_t>(thresholds.getLth() * noise.getNoiseFast(strip, detNoiseRange) + 0.5f);

    assert(lowThresh >= 0);
    assert(lowThresh <= highThresh);

    stat[i] = zeroTh;
    if (ladc >= lowThresh)
      stat[i] = lowTh;
    if (ladc >= highThresh)
      stat[i] = highTh;
  }

  for (size_t i = 0; i < inSize; i++) {
    auto strip = (uint32_t)in[i].strip();
    Payload ldata;

    ldata.stat = stat[i];
    ldata.statPrev = zeroTh;
    ldata.statNext = zeroTh;
    ldata.statPrev2 = zeroTh;
    ldata.statNext2 = zeroTh;

    if (((strip) % 128) == 127) {
      ldata.statNext = zeroTh;
    } else if (i + 1 < inSize && in[i + 1].strip() == strip + 1) {
      ldata.statNext = stat[i + 1];
      if (((strip) % 128) == 126) {
        ldata.statNext2 = zeroTh;
      } else if (i + 2 < inSize && in[i + 2].strip() == strip + 2) {
        ldata.statNext2 = stat[i + 2];
      }
    }

    if (((strip) % 128) == 0) {
      ldata.statPrev = zeroTh;
    } else if (i >= 1 && in[i - 1].strip() == strip - 1) {
      ldata.statPrev = stat[i - 1];
      if (((strip) % 128) == 1) {
        ldata.statPrev2 = zeroTh;
      } else if (i >= 2 && in[i - 2].strip() == strip - 2) {
        ldata.statPrev2 = stat[i - 2];
      }
    }

    ldata.statMaxNeigh = std::max(ldata.statPrev, ldata.statNext);

    if (isDigiValid(ldata, theFEDalgorithm)) {
      selectedSignal.push_back(SiStripDigi(strip, in[i].adc()));
    }
  }
}

void SiStripFedZeroSuppression::suppress(const edm::DetSet<SiStripRawDigi>& in, edm::DetSet<SiStripDigi>& out) {
  const uint32_t detID = out.id;
  SiStripNoises::Range detNoiseRange = noise_->getRange(detID);
  SiStripThreshold::Range detThRange = threshold_->getRange(detID);
#ifdef DEBUG_SiStripZeroSuppression_
  if (edm::isDebugEnabled())
    LogTrace("SiStripZeroSuppression")
        << "[SiStripFedZeroSuppression::suppress] Zero suppression on edm::DetSet<SiStripRawDigi>: detID " << detID
        << " size = " << in.data.size();
#endif
  edm::DetSet<SiStripRawDigi>::const_iterator in_iter = in.data.begin();
  for (; in_iter != in.data.end(); in_iter++) {
    const uint32_t strip = (uint32_t)(in_iter - in.data.begin());

#ifdef DEBUG_SiStripZeroSuppression_
    if (edm::isDebugEnabled())
      LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] detID= " << detID
                                         << " strip= " << strip << "  adc= " << in_iter->adc();
#endif
    adc = in_iter->adc();

    SiStripThreshold::Data thresholds = threshold_->getData(strip, detThRange);
    theFEDlowThresh = static_cast<int16_t>(thresholds.getLth() * noise_->getNoiseFast(strip, detNoiseRange) + 0.5);
    theFEDhighThresh = static_cast<int16_t>(thresholds.getHth() * noise_->getNoiseFast(strip, detNoiseRange) + 0.5);

    adcPrev = -9999;
    adcNext = -9999;
    /*
      If a strip is the last one on the chip
      set its next neighbor's thresholds to infinity
      because the FED does not merge clusters across
      chip boundaries right now
    */
    if (strip % 128 == 127) {
      adcNext = 0;
      theNextFEDlowThresh = 9999;
      theNextFEDhighThresh = 9999;
    } else {
      adcNext = (in_iter + 1)->adc();
      SiStripThreshold::Data thresholds_1 = threshold_->getData(strip + 1, detThRange);
      theNextFEDlowThresh =
          static_cast<int16_t>(thresholds_1.getLth() * noise_->getNoiseFast(strip + 1, detNoiseRange) + 0.5);
      theNextFEDhighThresh =
          static_cast<int16_t>(thresholds_1.getHth() * noise_->getNoiseFast(strip + 1, detNoiseRange) + 0.5);
    }
    /*
      Similarily, for the first strip 
      on a chip
    */
    if (strip % 128 == 0) {
      adcPrev = 0;
      thePrevFEDlowThresh = 9999;
      thePrevFEDhighThresh = 9999;
    } else {
      adcPrev = (in_iter - 1)->adc();
      SiStripThreshold::Data thresholds_1 = threshold_->getData(strip - 1, detThRange);
      thePrevFEDlowThresh =
          static_cast<int16_t>(thresholds_1.getLth() * noise_->getNoiseFast(strip - 1, detNoiseRange) + 0.5);
      thePrevFEDhighThresh =
          static_cast<int16_t>(thresholds_1.getHth() * noise_->getNoiseFast(strip - 1, detNoiseRange) + 0.5);
    }
    if (adcNext < adcPrev) {
      adcMaxNeigh = adcPrev;
      theNeighFEDlowThresh = thePrevFEDlowThresh;
      theNeighFEDhighThresh = thePrevFEDhighThresh;
    } else {
      adcMaxNeigh = adcNext;
      theNeighFEDlowThresh = theNextFEDlowThresh;
      theNeighFEDhighThresh = theNextFEDhighThresh;
    }

    //Find adc values for next neighbouring strips
    adcPrev2 = -9999;
    adcNext2 = -9999;
    thePrev2FEDlowThresh = 1;
    theNext2FEDlowThresh = 1;
    if (strip % 128 >= 126) {
      adcNext2 = 0;
      theNext2FEDlowThresh = 9999;
    } else if (strip % 128 < 126) {
      adcNext2 = (in_iter + 2)->adc();
      theNext2FEDlowThresh = static_cast<int16_t>(
          threshold_->getData(strip + 2, detThRange).getLth() * noise_->getNoiseFast(strip + 2, detNoiseRange) + 0.5);
    }
    if (strip % 128 <= 1) {
      adcPrev2 = 0;
      thePrev2FEDlowThresh = 9999;
    } else if (strip % 128 > 1) {
      adcPrev2 = (in_iter - 2)->adc();
      thePrev2FEDlowThresh = static_cast<int16_t>(
          threshold_->getData(strip - 2, detThRange).getLth() * noise_->getNoiseFast(strip - 2, detNoiseRange) + 0.5);
    }
    //GB 23/6/08: truncation should be done at the very beginning
    if (isAValidDigi())
      out.data.push_back(SiStripDigi(strip, truncate(in_iter->adc())));
  }
}

void SiStripFedZeroSuppression::fillThresholds_(const uint32_t detID, size_t size) {
  SiStripNoises::Range detNoiseRange = noise_->getRange(detID);
  SiStripThreshold::Range detThRange = threshold_->getRange(detID);

  if (highThr_.size() != size) {
    highThr_.resize(size);
    lowThr_.resize(size);
    noises_.resize(size);
    highThrSN_.resize(size);
    lowThrSN_.resize(size);
  }

  noise_->allNoises(noises_, detNoiseRange);
  threshold_->allThresholds(lowThrSN_, highThrSN_, detThRange);  // thresholds as S/N
  for (size_t strip = 0; strip < size; ++strip) {
    float noise = noises_[strip];
    //  uncomment line below to check bluk noise decoding
    //assert( noise == noiseHandle->getNoiseFast(strip,detNoiseRange) );
    highThr_[strip] = static_cast<int16_t>(highThrSN_[strip] * noise + 0.5 + 1e-6);
    lowThr_[strip] = static_cast<int16_t>(lowThrSN_[strip] * noise + 0.5 + 1e-6);
    // Note: it's a bit wierd, but there are some cases for which 'highThrSN_[strip]*noise' is an exact integer
    //   but due to roundoffs it gets rounded to the integer below if.
    //   Apparently the optimized code inlines differently and this changes the roundoff.
    //   The +1e-6 fixes the problem.   [GPetruc]
  }
}

void SiStripFedZeroSuppression::suppress(const std::vector<int16_t>& in,
                                         uint16_t firstAPV,
                                         edm::DetSet<SiStripDigi>& out) {
  const uint32_t detID = out.id;
  size_t size = in.size();
#ifdef DEBUG_SiStripZeroSuppression_
  if (edm::isDebugEnabled())
    LogTrace("SiStripZeroSuppression")
        << "[SiStripFedZeroSuppression::suppress] Zero suppression on std::vector<int16_t>: detID " << detID
        << " size = " << in.size();
#endif

  fillThresholds_(detID, size + firstAPV * 128);  // want to decouple this from the other cost

  std::vector<int16_t>::const_iterator in_iter = in.begin();
  uint16_t strip = firstAPV * 128;
  for (; strip < size + firstAPV * 128; ++strip, ++in_iter) {
    size_t strip_mod_128 = strip & 127;
#ifdef DEBUG_SiStripZeroSuppression_
    if (edm::isDebugEnabled())
      LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress]  detID= " << detID
                                         << " strip= " << strip << "  adc= " << *in_iter;
#endif
    adc = *in_iter;

    theFEDlowThresh = lowThr_[strip];
    theFEDhighThresh = highThr_[strip];

    //Find adc values for neighbouring strips

    /*
      If a strip is the last one on the chip
      set its next neighbor's thresholds to infinity
      because the FED does not merge clusters across
      chip boundaries right now
    */

    //adcPrev = -9999;  // useless, they are set
    //adcNext = -9999;  // in the next lines in any case
    if (strip_mod_128 == 127) {
      adcNext = 0;
      theNextFEDlowThresh = 9999;
      theNextFEDhighThresh = 9999;
    } else {
      adcNext = *(in_iter + 1);
      theNextFEDlowThresh = lowThr_[strip + 1];
      theNextFEDhighThresh = highThr_[strip + 1];
    }

    /*
      Similarily, for the first strip 
      on a chip
    */
    if (strip_mod_128 == 0) {
      adcPrev = 0;
      thePrevFEDlowThresh = 9999;
      thePrevFEDhighThresh = 9999;
    } else {
      adcPrev = *(in_iter - 1);
      thePrevFEDlowThresh = lowThr_[strip - 1];
      thePrevFEDhighThresh = highThr_[strip - 1];
    }

    if (adcNext < adcPrev) {
      adcMaxNeigh = adcPrev;
      theNeighFEDlowThresh = thePrevFEDlowThresh;
      theNeighFEDhighThresh = thePrevFEDhighThresh;
    } else {
      adcMaxNeigh = adcNext;
      theNeighFEDlowThresh = theNextFEDlowThresh;
      theNeighFEDhighThresh = theNextFEDhighThresh;
    }

    //Find adc values for next neighbouring strips
    //adcPrev2 = -9999;           //
    //adcNext2 = -9999;           // useless to set them here
    //thePrev2FEDlowThresh  = 1;  // they are overwritten always in the next 8 lines
    //theNext2FEDlowThresh  = 1;  //
    if (strip_mod_128 >= 126) {
      adcNext2 = 0;
      theNext2FEDlowThresh = 9999;
      //} else if ( strip_mod_128 < 126 ) { // if it's not >= then is <, no need to "if" again
    } else {
      adcNext2 = *(in_iter + 2);
      theNext2FEDlowThresh = lowThr_[strip + 2];
    }
    if (strip_mod_128 <= 1) {
      adcPrev2 = 0;
      thePrev2FEDlowThresh = 9999;
      //} else if ( strip_mod_128 > 1 ) { // same as above
    } else {
      adcPrev2 = *(in_iter - 2);
      thePrev2FEDlowThresh = lowThr_[strip - 2];
      ;
    }

    if (isAValidDigi()) {
#ifdef DEBUG_SiStripZeroSuppression_
      if (edm::isDebugEnabled())
        LogTrace("SiStripZeroSuppression")
            << "[SiStripFedZeroSuppression::suppress] DetId " << out.id << " strip " << strip << " adc " << *in_iter
            << " digiCollection size " << out.data.size();
#endif
      //GB 23/6/08: truncation should be done at the very beginning
      out.push_back(SiStripDigi(strip, (*in_iter < 0 ? 0 : truncate(*in_iter))));
    }
  }
}

bool SiStripFedZeroSuppression::isAValidDigi() {
#ifdef DEBUG_SiStripZeroSuppression_

  if (edm::isDebugEnabled()) {
    LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] "
                                       << "\n\t adc " << adc << "\n\t adcPrev " << adcPrev << "\n\t adcNext " << adcNext
                                       << "\n\t adcMaxNeigh " << adcMaxNeigh << "\n\t adcPrev2 " << adcPrev2
                                       << "\n\t adcNext2 " << adcNext2 << std::endl;

    LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] "
                                       << "\n\t theFEDlowThresh " << theFEDlowThresh << "\n\t theFEDhighThresh "
                                       << theFEDhighThresh << "\n\t thePrevFEDlowThresh " << thePrevFEDlowThresh
                                       << "\n\t thePrevFEDhighThresh " << thePrevFEDhighThresh
                                       << "\n\t theNextFEDlowThresh " << theNextFEDlowThresh
                                       << "\n\t theNextFEDhighThresh " << theNextFEDhighThresh
                                       << "\n\t theNeighFEDlowThresh " << theNeighFEDlowThresh
                                       << "\n\t theNeighFEDhighThresh " << theNeighFEDhighThresh
                                       << "\n\t thePrev2FEDlowThresh " << thePrev2FEDlowThresh
                                       << "\n\t theNext2FEDlowThresh " << theNext2FEDlowThresh << std::endl;
  }
#endif
  // Decide if this strip should be accepted.
  bool accept = false;
  switch (theFEDalgorithm) {
    case 1:
      accept = (adc >= theFEDlowThresh);
      break;
    case 2:
      accept = (adc >= theFEDhighThresh || (adc >= theFEDlowThresh && adcMaxNeigh >= theNeighFEDlowThresh));
      break;
    case 3:
      accept = (adc >= theFEDhighThresh || (adc >= theFEDlowThresh && adcMaxNeigh >= theNeighFEDhighThresh));
      break;
    case 4:
      accept =
          ((adc >= theFEDhighThresh)     //Test for adc>highThresh (same as algorithm 2)
           || ((adc >= theFEDlowThresh)  //Test for adc>lowThresh, with neighbour adc>lowThresh (same as algorithm 2)
               && (adcMaxNeigh >= theNeighFEDlowThresh)) ||
           ((adc < theFEDlowThresh)                 //Test for adc<lowThresh
            && (((adcPrev >= thePrevFEDhighThresh)  //with both neighbours>highThresh
                 && (adcNext >= theNextFEDhighThresh)) ||
                ((adcPrev >= thePrevFEDhighThresh)    //OR with previous neighbour>highThresh and
                 && (adcNext >= theNextFEDlowThresh)  //both the next neighbours>lowThresh
                 && (adcNext2 >= theNext2FEDlowThresh)) ||
                ((adcNext >= theNextFEDhighThresh)    //OR with next neighbour>highThresh and
                 && (adcPrev >= thePrevFEDlowThresh)  //both the previous neighbours>lowThresh
                 && (adcPrev2 >= thePrev2FEDlowThresh)) ||
                ((adcNext >= theNextFEDlowThresh)       //OR with both next neighbours>lowThresh and
                 && (adcNext2 >= theNext2FEDlowThresh)  //both the previous neighbours>lowThresh
                 && (adcPrev >= thePrevFEDlowThresh) && (adcPrev2 >= thePrev2FEDlowThresh)))));
      break;
    case 5:
      accept = adc > 0;
      break;
  }
  return accept;
}
