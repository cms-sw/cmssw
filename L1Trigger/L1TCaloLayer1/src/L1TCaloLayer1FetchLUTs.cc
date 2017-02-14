// This function fetches Layer1 ECAL and HCAL LUTs from CMSSW configuration
// It is provided as a global helper function outside of class structure
// so that it can be shared by L1CaloLayer1 and L1CaloLayer1Spy

#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "L1TCaloLayer1FetchLUTs.hh"
#include "UCTLogging.hh"

bool L1TCaloLayer1FetchLUTs(const edm::EventSetup& iSetup, 
			    std::vector< std::vector< std::vector< std::vector < uint32_t > > > > &eLUT,
			    std::vector< std::vector< std::vector< std::vector < uint32_t > > > > &hLUT,
                            std::vector< std::vector< std::vector< uint32_t > > > &hfLUT,
			    bool useLSB,
			    bool useCalib,
			    bool useECALLUT,
			    bool useHCALLUT,
                            bool useHFLUT) {

  int hfValid = 1;
  edm::ESHandle<HcalTrigTowerGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  if (! pG->use1x1()){
    edm::LogError("L1TCaloLayer1FetchLUTs") << "Using Stage2-Layer1 but HCAL Geometry has use1x1 = 0! HF will be suppressed.  Check Global Tag, etc.";
    hfValid = 0;
  } 

  // CaloParams contains all persisted parameters for Layer 1
  edm::ESHandle<l1t::CaloParams> paramsHandle;
  iSetup.get<L1TCaloStage2ParamsRcd>().get(paramsHandle);
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

  // ECal/HCal scale factors will be a x*y*56 array:
  //   56 eta scale factors (-28 .. -1 1 .. 28)
  //   x = size of Real ET Bins vector
  //   y = size of Real Phi Bins vector
  //   So, index = phiBin*etBin*56+etBin*56+ieta
  auto ecalScaleETBins = caloParams.layer1ECalScaleETBins();
  auto ecalScalePhiBins = caloParams.layer1ECalScalePhiBins();
  auto ecalSF = caloParams.layer1ECalScaleFactors();
  if ( ecalSF.size() != ecalScaleETBins.size()*ecalScalePhiBins.size()*56 ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1ECalScaleFactors().size() != caloParams.layer1ECalScaleETBins().size()*caloParams.layer1ECalScalePhiBins.size()*56 !!";
    return false;
  }
  auto hcalScaleETBins = caloParams.layer1HCalScaleETBins();
  auto hcalScalePhiBins = caloParams.layer1HCalScalePhiBins();
  auto hcalSF = caloParams.layer1HCalScaleFactors();
  if ( hcalSF.size() != hcalScaleETBins.size()*hcalScalePhiBins.size()*56 ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HCalScaleFactors().size() != caloParams.layer1HCalScaleETBins().size()*caloParams.layer1HCalScalePhiBins.size()*56 !!";
    return false;
  }

  // HF 1x1 scale factors will be a x*y*24 array:
  //  24 eta scale factors (-41 .. -30 30 .. 41)
  //   x = size of Real ET Bins vector
  //   x = size of Real Phi Bins vector
  //  So, index = etBin*12+ietaHF
  auto hfScaleETBins = caloParams.layer1HFScaleETBins();
  auto hfScalePhiBins = caloParams.layer1HFScalePhiBins();
  auto hfSF = caloParams.layer1HFScaleFactors();
  if ( hfSF.size() != hfScaleETBins.size()*hfScalePhiBins.size()*24 ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "caloParams.layer1HFScaleFactors().size() != caloParams.layer1HFScaleETBins().size()*caloParams.layer1HFScalePhiBins.size()*24 !!";
    return false;
  }

  // Sanity check scale factors exist
  if ( useCalib && (ecalSF.size()==0 || hcalSF.size()==0 || hfSF.size()==0) ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "Layer 1 calibrations requested (useCalib = True) but there are missing scale factors in CaloParams!  Please check conditions setup.";
    return false;
  }
  // get energy scale to convert input from ECAL - this should be linear with LSB = 0.5 GeV
  const double ecalLSB = 0.5;
      
  // get energy scale to convert input from HCAL
  edm::ESHandle<CaloTPGTranscoder> decoder;
  iSetup.get<CaloTPGRecord>().get(decoder);
  if ( decoder.product() == nullptr ) {
    edm::LogError("L1TCaloLayer1FetchLUTs") << "Missing CaloTPGTranscoder object! Check Global Tag, etc.";
    return false;
  }

  // TP compression scale is always phi symmetric
  // We default to 3 since HF has no ieta=41 iphi=1,2
  auto decodeHcalEt = [&decoder](int iEta, uint32_t compressedEt, uint32_t iPhi=3) -> double {
    HcalTriggerPrimitiveSample sample(compressedEt);
    HcalTrigTowerDetId id(iEta, iPhi);
    if ( std::abs(iEta) >= 30 ) {
      id.setVersion(1);
    }
    return decoder->hcaletValue(id, sample);
  };


  // Make ECal LUT
  for(int caloEta = -28; caloEta < 28; caloEta++) {
    uint32_t iEta = caloEta+28;
    for(uint32_t fb = 0; fb < 2; fb++) {
      for(uint32_t iPhi=0; iPhi<72; iPhi++) {
        for(uint32_t ecalInput = 0; ecalInput <= 0xFF; ecalInput++) {
	  uint32_t value = ecalInput;
	  if(useECALLUT) {
	    double linearizedECalInput = ecalInput*ecalLSB; // in GeV

            uint32_t etBin = 0;
            for(; etBin < ecalScaleETBins.size(); etBin++) {
              if(linearizedECalInput < ecalScaleETBins[etBin]) break;
            }
            if ( etBin >= ecalScaleETBins.size() ) etBin = ecalScaleETBins.size()-1;

            uint32_t phiBin = 0;
            for(; phiBin < ecalScalePhiBins.size(); phiBin++) {
              if(iPhi < ecalScalePhiBins[phiBin]) break;
            }
            if (phiBin >= ecalScalePhiBins.size()) phiBin = 0; // wrap in phi

            double calibratedECalInput = linearizedECalInput;
            if (useCalib) calibratedECalInput *= ecalSF.at(phiBin*etBin*56 + etBin*56 + iEta);
            if (useLSB) calibratedECalInput /= caloLSB;

	    value = calibratedECalInput;
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
	  eLUT[iPhi][iEta][fb][ecalInput] = value;
          }
      }
    }
  }

  // Make HCal LUT
  for(int caloEta = -28; caloEta < 28; caloEta++) {
    uint32_t iEta = caloEta+28;
    if (caloEta>=0) caloEta+=1;
    for(uint32_t fb = 0; fb < 2; fb++) {
      for(uint32_t iPhi=0; iPhi<72; iPhi++) {
        for(uint32_t hcalInput = 0; hcalInput <= 0xFF; hcalInput++) {
          uint32_t value = hcalInput;
          if(useHCALLUT) {
            // hcaletValue defined in L137 of CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.cc
            double linearizedHcalInput = decodeHcalEt(caloEta, hcalInput); // in GeV
            
            uint32_t etBin = 0;
            for(; etBin < hcalScaleETBins.size(); etBin++) {
              if(linearizedHcalInput < hcalScaleETBins[etBin]) break;
            }
            if ( etBin >= hcalScaleETBins.size() ) etBin = hcalScaleETBins.size()-1;

            uint32_t phiBin = 0;
            for(; phiBin < hcalScalePhiBins.size(); phiBin++) {
              if(iPhi < hcalScalePhiBins[phiBin]) break;
            }
            if (phiBin >= hcalScalePhiBins.size()) phiBin = 0; // wrap in phi

            double calibratedHcalInput = linearizedHcalInput;
            if(useCalib) calibratedHcalInput *= hcalSF.at(phiBin*etBin*56 + etBin*56 + iEta);
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
          hLUT[iPhi][iEta][fb][hcalInput] = value;
        }
      }
    }
  }

  // Make HF LUT
  for(uint32_t etaHF = -12; etaHF < 12; etaHF++) {
    uint32_t etaBin = etaHF+12;
    uint32_t caloEta = etaHF>=0 ? etaHF+31 : etaHF-30;
    for(uint32_t iPhi=0; iPhi<72; iPhi++) {
      for(uint32_t etCode = 0; etCode < 256; etCode++) {
        uint32_t value = etCode;
        if(useHFLUT) {
          
          double linearizedHFInput = 0;
          if (hfValid){
            linearizedHFInput = decodeHcalEt(caloEta, value); // in GeV
          }

          uint32_t etBin = 0;
          for(; etBin < hfScaleETBins.size(); etBin++) {
            if(linearizedHFInput < hfScaleETBins[etBin]) break;
          }
          if ( etBin >= hfScaleETBins.size() ) etBin = hfScaleETBins.size()-1;

          uint32_t phiBin = 0;
          for(; phiBin < hfScalePhiBins.size(); phiBin++) {
            if(iPhi < hfScalePhiBins[phiBin]) break;
          }
          if (phiBin >= hfScalePhiBins.size()) phiBin = 0; // wrap in phi

          double calibratedHFInput = linearizedHFInput;
          if(useCalib) calibratedHFInput *= hfSF.at(phiBin*etBin*24+etBin*24+etaBin);
          if(useLSB) calibratedHFInput /= caloLSB;

          value = calibratedHFInput;
          if(value > 0xFF) {
            value = 0xFF;
          }
        }
        hfLUT[iPhi][etaBin][etCode] = value;
      }
    }
  }
  return true;
}
/* vim: set ts=8 sw=2 tw=0 et :*/
