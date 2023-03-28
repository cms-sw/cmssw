// This function fetches Layer1 ECAL and HCAL LUTs from CMSSW configuration
// It is provided as a global helper function outside of class structure
// so that it can be shared by L1CaloLayer1 and L1CaloLayer1Spy

#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "L1TCaloLayer1FetchLUTs.hh"
#include "UCTLogging.hh"

using namespace l1tcalo;

bool L1TCaloLayer1FetchLUTs(
    const L1TCaloLayer1FetchLUTsTokens &iTokens,
    const edm::EventSetup &iSetup,
    std::vector<std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> > &eLUT,
    std::vector<std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> > &hLUT,
    std::vector<std::array<std::array<uint32_t, nEtBins>, nHfEtaBins> > &hfLUT,
    std::vector<unsigned long long int> &hcalFBLUT,
    std::vector<unsigned int> &ePhiMap,
    std::vector<unsigned int> &hPhiMap,
    std::vector<unsigned int> &hfPhiMap,
    bool useLSB,
    bool useCalib,
    bool useECALLUT,
    bool useHCALLUT,
    bool useHFLUT,
    bool useHCALFBLUT,
    int fwVersion) {
  int hfValid = 1;
  const HcalTrigTowerGeometry &pG = iSetup.getData(iTokens.geom_);
  if (!pG.use1x1()) {
    edm::LogError("L1TCaloLayer1FetchLUTs")
        << "Using Stage2-Layer1 but HCAL Geometry has use1x1 = 0! HF will be suppressed.  Check Global Tag, etc.";
    hfValid = 0;
  }

  // CaloParams contains all persisted parameters for Layer 1
  edm::ESHandle<l1t::CaloParams> paramsHandle = iSetup.getHandle(iTokens.params_);
  if (not paramsHandle.isValid()) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "Missing CaloParams object! Check Global Tag, etc.";
    return false;
  }
  l1t::CaloParamsHelper caloParams(*paramsHandle.product());

  // Calo Trigger Layer1 output LSB Real ET value
  double caloLSB = caloParams.towerLsbSum();
  if (caloLSB != 0.5) {
    // Lots of things expect this, better give fair warning if not
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloLSB (caloParams.towerLsbSum()) != 0.5, actually = " << caloLSB;
  }

  // ECal/HCal scale factors will be a x*y*28 array:
  //   ieta = 28 eta scale factors (1 .. 28)
  //   etBin = size of Real ET Bins vector
  //   phiBin = max(Real Phi Bins vector)
  //   So, index = phiBin*etBin*28+etBin*28+ieta
  auto ecalScaleETBins = caloParams.layer1ECalScaleETBins();
  auto ecalScalePhiBins = caloParams.layer1ECalScalePhiBins();
  if (ecalScalePhiBins.empty()) {
    // Backwards-compatibility (no phi binning)
    ecalScalePhiBins.resize(36, 0);
  } else if (ecalScalePhiBins.size() % 36 != 0) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1ECalScaleETBins().size() is not multiple of 36 !!";
    return false;
  }
  size_t numEcalPhiBins = (*std::max_element(ecalScalePhiBins.begin(), ecalScalePhiBins.end())) + 1;
  auto ecalSF = caloParams.layer1ECalScaleFactors();
  if (ecalSF.size() != ecalScaleETBins.size() * numEcalPhiBins * 28) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1ECalScaleFactors().size() != "
                                               "caloParams.layer1ECalScaleETBins().size()*numEcalPhiBins*28 !!";
    return false;
  }
  auto hcalScaleETBins = caloParams.layer1HCalScaleETBins();
  auto hcalScalePhiBins = caloParams.layer1HCalScalePhiBins();
  if (hcalScalePhiBins.empty()) {
    hcalScalePhiBins.resize(36, 0);
  } else if (hcalScalePhiBins.size() % 36 != 0) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HCalScaleETBins().size() is not multiple of 36 !!";
    return false;
  }
  size_t numHcalPhiBins = (*std::max_element(hcalScalePhiBins.begin(), hcalScalePhiBins.end())) + 1;
  auto hcalSF = caloParams.layer1HCalScaleFactors();
  if (hcalSF.size() != hcalScaleETBins.size() * numHcalPhiBins * 28) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HCalScaleFactors().size() != "
                                               "caloParams.layer1HCalScaleETBins().size()*numHcalPhiBins*28 !!";
    return false;
  }

  // HF 1x1 scale factors will be a x*y*12 array:
  //   ieta = 12 eta scale factors (30 .. 41)
  //   etBin = size of Real ET Bins vector
  //   phiBin = max(Real Phi Bins vector)
  //   So, index = phiBin*etBin*12+etBin*12+ieta
  auto hfScaleETBins = caloParams.layer1HFScaleETBins();
  auto hfScalePhiBins = caloParams.layer1HFScalePhiBins();
  if (hfScalePhiBins.empty()) {
    hfScalePhiBins.resize(36, 0);
  } else if (hfScalePhiBins.size() % 36 != 0) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HFScaleETBins().size() is not multiple of 36 !!";
    return false;
  }
  size_t numHFPhiBins = (*std::max_element(hfScalePhiBins.begin(), hfScalePhiBins.end())) + 1;
  auto hfSF = caloParams.layer1HFScaleFactors();
  if (hfSF.size() != hfScaleETBins.size() * numHFPhiBins * 12) {
    edm::LogError("L1TCaloLayer1FetchLUTs")
        << "caloParams.layer1HFScaleFactors().size() != caloParams.layer1HFScaleETBins().size()*numHFPhiBins*12 !!";
    return false;
  }

  // HCAL FB LUT will be a 1*28 array:
  //   ieta = 28 eta scale factors (1 .. 28)
  //   So, index = ieta
  auto fbLUTUpper = caloParams.layer1HCalFBLUTUpper();
  auto fbLUTLower = caloParams.layer1HCalFBLUTLower();
  if (fbLUTUpper.size() != 28) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HCalFBLUTUpper().size() != 28 !!";
    return false;
  }
  if (fbLUTLower.size() != 28) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HCalFBLUTLower().size() != 28 !!";
    return false;
  }

  // Sanity check scale factors exist
  if (useCalib && (ecalSF.empty() || hcalSF.empty() || hfSF.empty() || fbLUTUpper.empty() || fbLUTLower.empty())) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "Layer 1 calibrations requested (useCalib = True) but there are missing "
                                               "scale factors in CaloParams!  Please check conditions setup.";
    return false;
  }
  // get energy scale to convert input from ECAL - this should be linear with LSB = 0.5 GeV
  const double ecalLSB = 0.5;

  // get energy scale to convert input from HCAL
  edm::ESHandle<CaloTPGTranscoder> decoder = iSetup.getHandle(iTokens.decoder_);
  if (not decoder.isValid()) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "Missing CaloTPGTranscoder object! Check Global Tag, etc.";
    return false;
  }

  // TP compression scale is always phi symmetric
  // We default to 3 since HF has no ieta=41 iphi=1,2
  auto decodeHcalEt = [&decoder](int iEta, uint32_t compressedEt, uint32_t iPhi = 3) -> double {
    HcalTriggerPrimitiveSample sample(compressedEt);
    HcalTrigTowerDetId id(iEta, iPhi);
    if (std::abs(iEta) >= 30) {
      id.setVersion(1);
    }
    return decoder->hcaletValue(id, sample);
  };

  // Make ECal LUT
  for (uint32_t phiBin = 0; phiBin < numEcalPhiBins; phiBin++) {
    std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> phiLUT;
    eLUT.push_back(phiLUT);
    for (uint32_t etaBin = 0; etaBin < nCalEtaBins; etaBin++) {
      for (uint32_t fb = 0; fb < nCalSideBins; fb++) {
        for (uint32_t ecalInput = 0; ecalInput <= 0xFF; ecalInput++) {
          uint32_t value = ecalInput;
          if (useECALLUT) {
            double linearizedECalInput = ecalInput * ecalLSB;  // in GeV

            uint32_t etBin = 0;
            for (; etBin < ecalScaleETBins.size(); etBin++) {
              if (linearizedECalInput < ecalScaleETBins[etBin])
                break;
            }
            if (etBin >= ecalScaleETBins.size())
              etBin = ecalScaleETBins.size() - 1;

            double calibratedECalInput = linearizedECalInput;
            if (useCalib)
              calibratedECalInput *= ecalSF.at(phiBin * ecalScaleETBins.size() * 28 + etBin * 28 + etaBin);
            if (useLSB)
              calibratedECalInput /= caloLSB;

            value = calibratedECalInput;
            if (fwVersion > 2) {
              // Saturate if either decompressed value is over 127.5 GeV or input saturated
              // (meaningless for ecal, since ecalLSB == caloLSB)
              if (value > 0xFF || ecalInput == 0xFF) {
                value = 0xFF;
              }
            } else {
              if (value > 0xFF) {
                value = 0xFF;
              }
            }
          }
          if (value == 0) {
            value = (1 << 11);
          } else {
            uint32_t et_log2 = ((uint32_t)log2(value)) & 0x7;
            value |= (et_log2 << 12);
          }
          value |= (fb << 10);
          eLUT[phiBin][etaBin][fb][ecalInput] = value;
        }
      }
    }
  }

  // Make HCal LUT
  for (uint32_t phiBin = 0; phiBin < numHcalPhiBins; phiBin++) {
    std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> phiLUT;
    hLUT.push_back(phiLUT);
    for (uint32_t etaBin = 0; etaBin < nCalEtaBins; etaBin++) {
      int caloEta = etaBin + 1;
      int iPhi = 3;
      auto pos = std::find(hcalScalePhiBins.begin(), hcalScalePhiBins.end(), phiBin);
      if (pos != hcalScalePhiBins.end()) {
        // grab an iPhi bin
        auto index = std::distance(hcalScalePhiBins.begin(), pos);
        if (index < 18) {
          caloEta *= -1;
          iPhi = index * 4 + 1;
        } else {
          iPhi = (index - 18) * 4 + 1;
        }
      }
      for (uint32_t fb = 0; fb < nCalSideBins; fb++) {
        for (uint32_t hcalInput = 0; hcalInput <= 0xFF; hcalInput++) {
          uint32_t value = hcalInput;
          if (useHCALLUT) {
            // hcaletValue defined in L137 of CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.cc
            double linearizedHcalInput = decodeHcalEt(caloEta, hcalInput, iPhi);  // in GeV

            uint32_t etBin = 0;
            for (; etBin < hcalScaleETBins.size(); etBin++) {
              if (linearizedHcalInput < hcalScaleETBins[etBin])
                break;
            }
            if (etBin >= hcalScaleETBins.size())
              etBin = hcalScaleETBins.size() - 1;

            double calibratedHcalInput = linearizedHcalInput;
            if (useCalib)
              calibratedHcalInput *= hcalSF.at(phiBin * hcalScaleETBins.size() * 28 + etBin * 28 + etaBin);
            if (useLSB)
              calibratedHcalInput /= caloLSB;

            value = calibratedHcalInput;
            if (fwVersion > 2) {
              // Saturate if either decompressed value is over 127.5 GeV or input saturated
              if (value > 0xFF || hcalInput == 0xFF) {
                value = 0xFF;
              }
            } else {
              if (value > 0xFF) {
                value = 0xFF;
              }
            }
          }
          if (value == 0) {
            value = (1 << 11);
          } else {
            uint32_t et_log2 = ((uint32_t)log2(value)) & 0x7;
            value |= (et_log2 << 12);
          }
          value |= (fb << 10);
          hLUT[phiBin][etaBin][fb][hcalInput] = value;
        }
      }
    }
  }

  // Make HF LUT
  for (uint32_t phiBin = 0; phiBin < numHFPhiBins; phiBin++) {
    std::array<std::array<uint32_t, nEtBins>, nHfEtaBins> phiLUT;
    hfLUT.push_back(phiLUT);
    for (uint32_t etaBin = 0; etaBin < nHfEtaBins; etaBin++) {
      int caloEta = etaBin + 30;
      int iPhi = 3;
      auto pos = std::find(hfScalePhiBins.begin(), hfScalePhiBins.end(), phiBin);
      if (pos != hfScalePhiBins.end()) {
        auto index = std::distance(hfScalePhiBins.begin(), pos);
        if (index < 18) {
          caloEta *= -1;
          iPhi = index * 4 - 1;
        } else {
          iPhi = (index - 18) * 4 - 1;
        }
        if (iPhi < 0)
          iPhi = 71;
      }
      for (uint32_t etCode = 0; etCode < nEtBins; etCode++) {
        uint32_t value = etCode;
        if (useHFLUT) {
          double linearizedHFInput = 0;
          if (hfValid) {
            linearizedHFInput = decodeHcalEt(caloEta, value, iPhi);  // in GeV
          }

          uint32_t etBin = 0;
          for (; etBin < hfScaleETBins.size(); etBin++) {
            if (linearizedHFInput < hfScaleETBins[etBin])
              break;
          }
          if (etBin >= hfScaleETBins.size())
            etBin = hfScaleETBins.size() - 1;

          double calibratedHFInput = linearizedHFInput;
          if (useCalib)
            calibratedHFInput *= hfSF.at(phiBin * hfScalePhiBins.size() * 12 + etBin * 12 + etaBin);
          if (useLSB)
            calibratedHFInput /= caloLSB;

          if (fwVersion > 2) {
            uint32_t absCaloEta = std::abs(caloEta);
            if (absCaloEta > 29 && absCaloEta < 40) {
              // Divide by two (since two duplicate towers are sent)
              calibratedHFInput *= 0.5;
            } else if (absCaloEta == 40 || absCaloEta == 41) {
              // Divide by four
              calibratedHFInput *= 0.25;
            }
            value = calibratedHFInput;
            // Saturate if either decompressed value is over 127.5 GeV or input saturated
            if (value >= 0xFF || etCode == 0xFF) {
              value = 0x1FD;
            }
          } else {
            value = calibratedHFInput;
            if (value > 0xFF) {
              value = 0xFF;
            }
          }
        }
        hfLUT[phiBin][etaBin][etCode] = value;
      }
    }
  }

  // Make HCal FB LUT
  for (uint32_t etaBin = 0; etaBin < 28; etaBin++) {
    uint64_t value = (((uint64_t)fbLUTUpper.at(etaBin)) << 32) | fbLUTLower.at(etaBin);
    hcalFBLUT.push_back(value);
  }

  // plus/minus, 18 CTP7, 4 iPhi each
  for (uint32_t isPos = 0; isPos < 2; isPos++) {
    for (uint32_t iPhi = 1; iPhi <= 72; iPhi++) {
      uint32_t card = floor((iPhi + 1) / 4);
      if (card > 17)
        card -= 18;
      ePhiMap[isPos * 72 + iPhi - 1] = ecalScalePhiBins[isPos * 18 + card];
      hPhiMap[isPos * 72 + iPhi - 1] = hcalScalePhiBins[isPos * 18 + card];
      hfPhiMap[isPos * 72 + iPhi - 1] = hfScalePhiBins[isPos * 18 + card];
    }
  }

  return true;
}
/* vim: set ts=8 sw=2 tw=0 et :*/
