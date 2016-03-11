// This function fetches Layer1 ECAL and HCAL LUTs from CMSSW configuration
// It is provided as a global helper function outside of class structure
// so that it can be shared by L1CaloLayer1 and L1CaloLayer1Spy

#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

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


  edm::ESHandle<l1t::CaloParams> paramsHandle;
  iSetup.get<L1TCaloParamsRcd>().get(paramsHandle);
  if ( paramsHandle.product() == nullptr ) return false;
  l1t::CaloParamsHelper caloParams(*paramsHandle.product());

  // Global Calo Trigger LSB ET value
  double caloLSB = caloParams.towerLsbSum(); // Will probably always be 0.5

  // ECal/HCal scale factors will be a 9*28 array:
  //   28 eta scale factors (1-28)
  //   in 9 ET bins (10, 15, 20, 25, 30, 35, 40, 45, Max) N.B. Max effectively is 256*caloLSB
  //   So, index = etBin*28+ieta
  const std::vector<double> caloSFETBins{10., 15., 20., 25., 30., 35., 40., 45., 1.e6};
  auto ecalSF = caloParams.layer1ECalScaleFactors();
  if ( ecalSF.size() != 9*28 ) return false;
  auto hcalSF = caloParams.layer1HCalScaleFactors();
  if ( hcalSF.size() != 9*28 ) return false;

  // HF 1x1 scale factors will be a 5*12 array:
  //  12 eta scale factors (30-41)
  //  in 5 et bins (5, 20, 30, 50, 256)
  //  So, index = etBin*12+ietaHF
  const std::vector<uint32_t> hfSFETBins{5, 20, 30, 50, 256};
  auto hfSF = caloParams.layer1HFScaleFactors();
  if ( hfSF.size() != 12*5 ) return false;

  // get energy scale to convert input from ECAL - this should be linear with LSB = 0.5 GeV
  const double ecalLSB = 0.5;
      
  // get energy scale to convert input from HCAL - this should be Landsberg's E to ET etc non-linear conversion factors
  edm::ESHandle<CaloTPGTranscoder> decoder;
  iSetup.get<CaloTPGRecord>().get(decoder);
  if ( decoder.product() == nullptr ) return false;

  // Make ECal LUT
  for(int absCaloEta = 1; absCaloEta <= 28; absCaloEta++) {
    uint32_t iEta = absCaloEta - 1;
    for(uint32_t fb = 0; fb < 2; fb++) {
      for(uint32_t ecalInput = 0; ecalInput <= 0xFF; ecalInput++) {
	uint32_t value = ecalInput;
	if(useECALLUT) {
	  double linearizedECalInput = ecalInput*ecalLSB;
	  double calibratedECalInput = linearizedECalInput;

          if ( useECALLUT ) {
            uint32_t etBin = 0;
            for(; etBin < caloSFETBins.size(); etBin++) {
              if(linearizedECalInput < caloSFETBins[etBin]) break;
            }
            calibratedECalInput *= ecalSF.at(etBin*28 + iEta);
          }

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
	  double linearizedHcalInput = decoder->hcaletValue(iEta, 0, hcalInput);
	  if(linearizedHcalInput != decoder->hcaletValue(-iEta, 0, hcalInput)) {
	    std::cerr << "L1TCaloLayer1FetchLUTs - hcal scale factors are different for positive and negative eta ! :(" << std::endl;
	  }
	  double calibratedHcalInput = linearizedHcalInput;
          if ( useHCALLUT ) {
            uint32_t etBin = 0;
            for(; etBin < caloSFETBins.size(); etBin++) {
              if(linearizedHcalInput < caloSFETBins[etBin]) break;
            }
            calibratedHcalInput *= hcalSF.at(etBin*28 + iEta);
          }
	  if(useLSB) value = calibratedHcalInput / caloLSB;
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
      if(useHFLUT) {
	uint32_t etBin = 0;
	for(; etBin < hfSFETBins.size(); etBin++) {
	  if(etCode < hfSFETBins[etBin]) break;
	}
	hfLUT[etaBin][etCode] = etCode * hfSF.at(etBin*12 + etaBin);
      }
      else {
	hfLUT[etaBin][etCode] = etCode;
      }
    }
  }
  return true;
}
