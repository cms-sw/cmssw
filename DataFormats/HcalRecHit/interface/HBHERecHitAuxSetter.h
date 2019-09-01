#ifndef DataFormats_HcalRecHit_HBHERecHitAuxSetter_h_
#define DataFormats_HcalRecHit_HBHERecHitAuxSetter_h_

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"
#include "DataFormats/HcalRecHit/interface/CaloRecHitAuxSetter.h"

//
// Set rechit "auxiliary words" for the Phase 1 HBHE reco.
//
// The standars CaloRecHit "aux" word contains ADC values 0-3.
//
// The "auxHBHE" word of HBHERecHit contains ADC values 4-7.
//
// The "auxPhase1" word of HBHERecHit contains ADC values 8-9
// and other info, as specified by the masks and offsets below.
//
struct HBHERecHitAuxSetter {
  static const unsigned MASK_ADC = 0xffff;
  static const unsigned OFF_ADC = 0;

  // How many ADC values are actually stored
  static const unsigned MASK_NSAMPLES = 0xf;
  static const unsigned OFF_NSAMPLES = 16;

  // Which ADC corresponds to the sample of interest.
  static const unsigned MASK_SOI = 0xf;
  static const unsigned OFF_SOI = 20;

  // CAPID for the sample of interest.
  static const unsigned MASK_CAPID = 0x3;
  static const unsigned OFF_CAPID = 24;

  // Various status bits (pack bools from HBHEChannelInfo)
  static const unsigned OFF_TDC_TIME = 26;
  static const unsigned OFF_DROPPED = 27;
  static const unsigned OFF_LINK_ERR = 28;
  static const unsigned OFF_CAPID_ERR = 29;

  // Flag identifying combined "Plan 1" rechits
  static const unsigned OFF_COMBINED = 30;

  // Main function for setting the aux words.
  constexpr static void setAux(const HBHEChannelInfo& info, HBHERecHit* rechit);
};

constexpr void HBHERecHitAuxSetter::setAux(const HBHEChannelInfo& info, HBHERecHit* rechit) {
  using namespace CaloRecHitAuxSetter;

  uint32_t aux = 0, auxHBHE = 0, auxPhase1 = 0;

  // Pack ADC values
  unsigned nSamples = info.nSamples();
  if (nSamples > 10)
    nSamples = 10;
  for (unsigned i = 0; i < 4 && i < nSamples; ++i)
    setField(&aux, 0xff, i * 8, info.tsAdc(i));
  for (unsigned i = 4; i < 8 && i < nSamples; ++i)
    setField(&auxHBHE, 0xff, (i - 4) * 8, info.tsAdc(i));
  for (unsigned i = 8; i < nSamples; ++i)
    setField(&auxPhase1, 0xff, (i - 8) * 8, info.tsAdc(i));

  // Pack other fields
  setField(&auxPhase1, MASK_NSAMPLES, OFF_NSAMPLES, nSamples);
  unsigned soi = info.soi();
  if (soi > 10)
    soi = 10;
  setField(&auxPhase1, MASK_SOI, OFF_SOI, soi);
  setField(&auxPhase1, MASK_CAPID, OFF_CAPID, info.capid());

  // Pack status bits
  setBit(&auxPhase1, OFF_TDC_TIME, info.hasTimeInfo());
  setBit(&auxPhase1, OFF_DROPPED, info.isDropped());
  setBit(&auxPhase1, OFF_LINK_ERR, info.hasLinkError());
  setBit(&auxPhase1, OFF_CAPID_ERR, info.hasCapidError());

  // Copy the aux words into the rechit
  rechit->setAux(aux);
  rechit->setAuxHBHE(auxHBHE);
  rechit->setAuxPhase1(auxPhase1);
}

#endif  // DataFormats_HcalRecHit_HBHERecHitAuxSetter_h_
