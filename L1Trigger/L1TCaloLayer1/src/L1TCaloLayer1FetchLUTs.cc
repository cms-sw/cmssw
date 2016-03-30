// This function fetches Layer1 ECAL and HCAL LUTs from CMSSW configuration
// It is provided as a global helper function outside of class structure
// so that it can be shared by L1CaloLayer1 and L1CaloLayer1Spy

#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"

#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"

#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "L1TCaloLayer1FetchLUTs.hh"
#include "UCTLogging.hh"

bool L1TCaloLayer1FetchLUTs(const edm::EventSetup& iSetup, 
			    std::vector< std::vector< std::vector < uint32_t > > > &eLUT,
			    std::vector< std::vector< std::vector < uint32_t > > > &hLUT,
			    bool useLSB,
			    bool useECALLUT,
			    bool useHCALLUT) {

  // get rct parameters - these should contain Laura Dodd's tower-level scalefactors (ET, eta)

  edm::ESHandle<L1RCTParameters> rctParameters;
  iSetup.get<L1RCTParametersRcd>().get(rctParameters);
  const L1RCTParameters* rctParameters_ = rctParameters.product();
  if(rctParameters_ == 0) return false;


  // get energy scale to convert input from ECAL - this should be linear with LSB = 0.5 GeV
  edm::ESHandle<L1CaloEcalScale> ecalScale;
  iSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
  const L1CaloEcalScale* e = ecalScale.product();
  if(e == 0) return false;

      
  // get energy scale to convert input from HCAL - this should be Landsberg's E to ET etc non-linear conversion factors
  edm::ESHandle<L1CaloHcalScale> hcalScale;
  iSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
  const L1CaloHcalScale* h = hcalScale.product();
  if(h == 0) return false;

  for(int absCaloEta = 1; absCaloEta <= 28; absCaloEta++) {
    uint32_t iEta = absCaloEta - 1;
    for(uint32_t fb = 0; fb < 2; fb++) {
      for(uint32_t ecalInput = 0; ecalInput <= 0xFF; ecalInput++) {
	uint32_t value = ecalInput;
	if(useECALLUT) {
	  double linearizedECalInput = e->et(ecalInput, absCaloEta, 1);
	  if(linearizedECalInput != (e->et(ecalInput, absCaloEta, -1))) {
	    LOG_ERROR << "L1TCaloLayer1FetchLUTs - ecal scale factors are different for positive and negative eta ! :(" << std::endl;
	  }
	  // Use hcal = 0 to get ecal only energy but in RCT JetMET scale - should be 8-bit max
	  double calibratedECalInput = linearizedECalInput;
	  if(useECALLUT) calibratedECalInput = rctParameters_->JetMETTPGSum(linearizedECalInput, 0, absCaloEta);
	  if(useLSB) value = calibratedECalInput / rctParameters_->jetMETLSB();
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

  for(int absCaloEta = 1; absCaloEta <= 28; absCaloEta++) {
    uint32_t iEta = absCaloEta - 1;
    for(uint32_t fb = 0; fb < 2; fb++) {
      for(uint32_t hcalInput = 0; hcalInput <= 0xFF; hcalInput++) {
	uint32_t value = hcalInput;
	if(useHCALLUT) {
	  double linearizedHcalInput = h->et(hcalInput, absCaloEta, 1);
	  if(linearizedHcalInput != (h->et(hcalInput, absCaloEta, -1))) {
	    LOG_ERROR << "L1TCaloLayer1FetchLUTs - hcal scale factors are different for positive and negative eta ! :(" << std::endl;
	  }
	  // Use ecal = 0 to get hcal only energy but in RCT JetMET scale - should be 8-bit max
	  double calibratedHcalInput = linearizedHcalInput;
	  if(useHCALLUT) calibratedHcalInput = rctParameters_->JetMETTPGSum(0, linearizedHcalInput, absCaloEta);
	  if(useLSB) value = calibratedHcalInput / rctParameters_->jetMETLSB();
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
  
  return true;
  
}
