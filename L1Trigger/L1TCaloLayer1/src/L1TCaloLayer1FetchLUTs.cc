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

#include "L1TCaloLayer1FetchLUTs.hh"

bool L1TCaloLayer1FetchLUTs(const edm::EventSetup& iSetup, 
			    std::vector< std::vector< std::vector < uint32_t > > > &eLUT,
			    std::vector< std::vector< std::vector < uint32_t > > > &hLUT,
                            std::vector< std::vector< uint32_t > > &hfLUT,
			    bool useLSB,
			    bool useECALLUT,
			    bool useHCALLUT,
                            bool useHFLUT) {

  // CaloParams contains all persisted parameters for Layer 1
  edm::ESHandle<l1t::CaloParams> paramsHandle;
  iSetup.get<L1TCaloParamsRcd>().get(paramsHandle);
  if ( paramsHandle.product() == nullptr ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "Missing CaloParams object! Check Global Tag, etc.";
    return false;
  }
  l1t::CaloParamsHelper caloParams(*paramsHandle.product());

  // Calo Trigger Layer1 output LSB Real ET value
  double caloLSB = caloParams.towerLsbSum();
  if ( caloLSB != 0.5 ) {
    // Lots of things expect this, better give fair warning if not
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloLSB (caloParams.towerLsbSum()) != 0.5, actually = " << caloLSB;
  }

  // ECal/HCal scale factors will be a 9*28 array:
  //   28 eta scale factors (1-28)
  //   in 9 ET bins (10, 15, 20, 25, 30, 35, 40, 45, Max) N.B. Max effectively is 256*caloLSB
  //   So, index = etBin*28+ieta
  const std::vector<double> caloSFETBins{10., 15., 20., 25., 30., 35., 40., 45., 1.e6};
  auto ecalSF = caloParams.layer1ECalScaleFactors();
  if ( ecalSF.size() != 9*28 ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1ECalScaleFactors().size() != 9*28 !!";
    return false;
  }
  auto hcalSF = caloParams.layer1HCalScaleFactors();
  if ( hcalSF.size() != 9*28 ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HCalScaleFactors().size() != 9*28 !!";
    return false;
  }

  // HF 1x1 scale factors will be a 5*12 array:
  //  12 eta scale factors (30-41)
  //  in 5 REAL ET bins (5, 20, 30, 50, Max)
  //  So, index = etBin*12+ietaHF
  const std::vector<double> hfSFETBins{5., 20., 30., 50., 1.e6};
  auto hfSF = caloParams.layer1HFScaleFactors();
  if ( hfSF.size() != 12*5 ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HFScaleFactors().size() != 5*12 !!";
    return false;
  }

  // get energy scale to convert input from ECAL - this should be linear with LSB = 0.5 GeV
  const double ecalLSB = 0.5;
      
  // get energy scale to convert input from HCAL - this should be Landsberg's E to ET etc non-linear conversion factors
  edm::ESHandle<CaloTPGTranscoder> decoder;
  iSetup.get<CaloTPGRecord>().get(decoder);
  if ( decoder.product() == nullptr ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "Missing CaloTPGTranscoder object! Check Global Tag, etc.";
    return false;
  }

  // Make ECal LUT
  for(int absCaloEta = 1; absCaloEta <= 28; absCaloEta++) {
    uint32_t iEta = absCaloEta - 1;
    for(uint32_t fb = 0; fb < 2; fb++) {
      for(uint32_t ecalInput = 0; ecalInput <= 0xFF; ecalInput++) {
	uint32_t value = ecalInput;
	if(useECALLUT) {
	  double linearizedECalInput = ecalInput*ecalLSB;

          uint32_t etBin = 0;
          for(; etBin < caloSFETBins.size(); etBin++) {
            if(linearizedECalInput < caloSFETBins[etBin]) break;
          }
          double calibratedECalInput = linearizedECalInput*ecalSF.at(etBin*28 + iEta);

	  if(useLSB) value = calibratedECalInput / caloLSB;
	  if(value > 0xFF) {
	    value = 0xFF;
	  }
	}
	if(value == 0) {
	  value = (1 << 11);
	}
	else {
	  uint32_t et_log2 = ((uint32_t) log2(value)) & 0x7;
	  value |= (et_log2 << 12);
	}
	value |= (fb << 10);
	eLUT[iEta][fb][ecalInput] = value;
      }
    }
  }

  // Make HCal LUT
  for(int absCaloEta = 1; absCaloEta <= 28; absCaloEta++) {
    uint32_t iEta = absCaloEta - 1;
    for(uint32_t fb = 0; fb < 2; fb++) {
      for(uint32_t hcalInput = 0; hcalInput <= 0xFF; hcalInput++) {
	uint32_t value = hcalInput;
	if(useHCALLUT) {
          // hcaletValue defined in L137 of CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.cc
	  double linearizedHcalInput = decoder->hcaletValue(absCaloEta, hcalInput);
	  if(linearizedHcalInput != decoder->hcaletValue(-absCaloEta, hcalInput)) {
	    edm::LogError("L1TCaloLayer1FetchLUTs") << "L1TCaloLayer1FetchLUTs - hcal scale factors are different for positive and negative eta ! :(" << std::endl;
	  }

          uint32_t etBin = 0;
          for(; etBin < caloSFETBins.size(); etBin++) {
            if(linearizedHcalInput < caloSFETBins[etBin]) break;
          }
          double calibratedHcalInput = linearizedHcalInput*hcalSF.at(etBin*28 + iEta);

	  if(useLSB) calibratedHcalInput /= caloLSB;

          value = calibratedHcalInput;
	  if(value > 0xFF) {
	    value = 0xFF;
	  }
	}
	if(value == 0) {
	  value = (1 << 11);
	}
	else {
	  uint32_t et_log2 = ((uint32_t) log2(value)) & 0x7;
	  value |= (et_log2 << 12);
	}
	value |= (fb << 10);
	hLUT[iEta][fb][hcalInput] = value;
      }
    }
  }

  // Make HF LUT
  for(uint32_t etaBin = 0; etaBin < 12; etaBin++) {
    for(uint32_t etCode = 0; etCode < 256; etCode++) {
      uint32_t value = etCode;
      if(useHFLUT) {
        double linearizedHFInput = decoder->hcaletValue(30+etaBin, value);

	uint32_t etBin = 0;
	for(; etBin < hfSFETBins.size(); etBin++) {
	  if(linearizedHFInput < hfSFETBins[etBin]) break;
	}
        double calibratedHFInput = linearizedHFInput*hfSF.at(etBin*12+etaBin);

        if(useLSB) calibratedHFInput /= caloLSB;

        value = calibratedHFInput;
        if(value > 0xFF) {
          value = 0xFF;
        }
      }
      hfLUT[etaBin][etCode] = value;
    }
  }
  return true;
}
/* vim: set ts=8 sw=2 tw=0 et :*/
