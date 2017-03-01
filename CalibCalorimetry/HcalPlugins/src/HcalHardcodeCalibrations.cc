// -*- C++ -*-
// Original Author:  Fedor Ratnikov
//
//

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

#include "CondFormats/DataRecord/interface/HcalAllRcds.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HcalHardcodeCalibrations.h"

//#define DebugLog

// class decleration
//

using namespace cms;

namespace {

  std::vector<HcalGenericDetId> allCells (const HcalTopology& hcaltopology, bool killHE = false) {
    static std::vector<HcalGenericDetId> result;
    int maxDepthHB=hcaltopology.maxDepthHB();
    int maxDepthHE=hcaltopology.maxDepthHE();

#ifdef DebugLog
    std::cout << std::endl << "HcalHardcodeCalibrations:   maxDepthHB, maxDepthHE = " 
	      <<  maxDepthHB << ", " <<  maxDepthHE << std::endl;
#endif

    if (result.size () <= 0) {
      for (int eta = -HcalDetId::kHcalEtaMask2; 
           eta <= HcalDetId::kHcalEtaMask2; eta++) {
        for (int phi = 0; phi <= HcalDetId::kHcalPhiMask2; phi++) {
          for (int depth = 1; depth < maxDepthHB + maxDepthHE; depth++) {
            for (int det = 1; det <= HcalForward; det++) {
	      HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	      if( killHE && HcalEndcap == cell.subdetId() ) continue;
	      if (hcaltopology.valid(cell)) {
		result.push_back (cell);
#ifdef DebugLog
		std::cout << " HcalHardcodedCalibrations: det|eta|phi|depth = "
			  << det << "|" << eta << "|" << phi << "|"
			  << depth << std::endl;  
#endif
	      }
	    }
	  }
	}
      } 
      ZdcTopology zdctopology;
      HcalZDCDetId zcell;
      HcalZDCDetId::Section section  = HcalZDCDetId::EM;
      for(int depth= 1; depth < 6; depth++){
	zcell = HcalZDCDetId(section, true, depth);
	if(zdctopology.valid(zcell)) result.push_back(zcell);
	zcell = HcalZDCDetId(section, false, depth);
	if(zdctopology.valid(zcell)) result.push_back(zcell);     
      }
      section = HcalZDCDetId::HAD;
      for(int depth= 1; depth < 5; depth++){
	zcell = HcalZDCDetId(section, true, depth);
	if(zdctopology.valid(zcell)) result.push_back(zcell);
	zcell = HcalZDCDetId(section, false, depth);
	if(zdctopology.valid(zcell)) result.push_back(zcell);     
      }
      section = HcalZDCDetId::LUM;
      for(int depth= 1; depth < 3; depth++){
	zcell = HcalZDCDetId(section, true, depth);
	if(zdctopology.valid(zcell)) result.push_back(zcell);
	zcell = HcalZDCDetId(section, false, depth);
	if(zdctopology.valid(zcell)) result.push_back(zcell);     
      }
      section = HcalZDCDetId::RPD;
      for(int depth= 1; depth < 17; depth++){
	zcell = HcalZDCDetId(section, true, depth);
	if(zdctopology.valid(zcell)) result.push_back(zcell);
	zcell = HcalZDCDetId(section, false, depth);
	if(zdctopology.valid(zcell)) result.push_back(zcell);     
      }

      // HcalGenTriggerTower (HcalGenericSubdetector = 5) 
      // NASTY HACK !!!
      // - As no valid(cell) check found for HcalTrigTowerDetId 
      // to create HT cells (ieta=1-28, iphi=1-72)&(ieta=29-32, iphi=1,5,... 69)

      for (int vers=0; vers<=HcalTrigTowerDetId::kHcalVersMask; ++vers) {
        for (int depth=0; depth<=HcalTrigTowerDetId::kHcalDepthMask; ++depth) {
          for (int eta = -HcalTrigTowerDetId::kHcalEtaMask; 
               eta <= HcalTrigTowerDetId::kHcalEtaMask; eta++) {
            for (int phi = 1; phi <= HcalTrigTowerDetId::kHcalPhiMask; phi++) {
              HcalTrigTowerDetId cell(eta, phi,depth,vers); 
              if (hcaltopology.validHT(cell)) {
		result.push_back (cell);
#ifdef DebugLog
		std::cout << " HcalHardcodedCalibrations: eta|phi|depth|vers = "
			  << eta << "|" << phi << "|" << depth << "|" << vers
			  << std::endl;  
#endif
	      }
	    }
	  }
	}
      }
    }
    return result;
  }

}

HcalHardcodeCalibrations::HcalHardcodeCalibrations ( const edm::ParameterSet& iConfig ): 
	he_recalibration(0), hf_recalibration(0), setHEdsegm(false), setHBdsegm(false)
{
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::HcalHardcodeCalibrations->...";

  if ( iConfig.exists("GainWidthsForTrigPrims") ) 
    switchGainWidthsForTrigPrims = iConfig.getParameter<bool>("GainWidthsForTrigPrims");
  else  switchGainWidthsForTrigPrims = false;
  
  //DB helper preparation
  dbHardcode.setHB(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("hb")));
  dbHardcode.setHE(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("he")));
  dbHardcode.setHF(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("hf")));
  dbHardcode.setHO(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("ho")));
  dbHardcode.setHBUpgrade(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("hbUpgrade")));
  dbHardcode.setHEUpgrade(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("heUpgrade")));
  dbHardcode.setHFUpgrade(HcalHardcodeParameters(iConfig.getParameter<edm::ParameterSet>("hfUpgrade")));
  dbHardcode.useHBUpgrade(iConfig.getParameter<bool>("useHBUpgrade"));
  dbHardcode.useHEUpgrade(iConfig.getParameter<bool>("useHEUpgrade"));
  dbHardcode.useHFUpgrade(iConfig.getParameter<bool>("useHFUpgrade"));
  dbHardcode.useHOUpgrade(iConfig.getParameter<bool>("useHOUpgrade"));
  dbHardcode.testHFQIE10(iConfig.getParameter<bool>("testHFQIE10"));
  dbHardcode.testHEPlan1(iConfig.getParameter<bool>("testHEPlan1"));
  dbHardcode.setKillHE(iConfig.getParameter<bool>("killHE"));
  dbHardcode.setSiPMCharacteristics(iConfig.getParameter<std::vector<edm::ParameterSet>>("SiPMCharacteristics"));

  useLayer0Weight = iConfig.getParameter<bool>("useLayer0Weight");
  // HE and HF recalibration preparation
  iLumi=iConfig.getParameter<double>("iLumi");

  if( iLumi > 0.0 ) {
    bool he_recalib = iConfig.getParameter<bool>("HERecalibration");
    bool hf_recalib = iConfig.getParameter<bool>("HFRecalibration");
    if(he_recalib) {
      double cutoff = iConfig.getParameter<double>("HEreCalibCutoff"); 
      he_recalibration = new HERecalibration(iLumi,cutoff);
    }
    if(hf_recalib && !iConfig.getParameter<edm::ParameterSet>("HFRecalParameterBlock").empty())  hf_recalibration = new HFRecalibration(iConfig.getParameter<edm::ParameterSet>("HFRecalParameterBlock"));
    
#ifdef DebugLog
    std::cout << " HcalHardcodeCalibrations:  iLumi = " <<  iLumi << std::endl;
#endif
  }

  std::vector <std::string> toGet = iConfig.getUntrackedParameter <std::vector <std::string> > ("toGet");
  for(std::vector <std::string>::iterator objectName = toGet.begin(); objectName != toGet.end(); ++objectName ) {
    bool all = *objectName == "all";
#ifdef DebugLog
    std::cout << "Load parameters for " << *objectName << std::endl;
#endif
    if ((*objectName == "Pedestals") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::producePedestals);
      findingRecord <HcalPedestalsRcd> ();
    }
    if ((*objectName == "PedestalWidths") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::producePedestalWidths);
      findingRecord <HcalPedestalWidthsRcd> ();
    }
    if ((*objectName == "Gains") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceGains);
      findingRecord <HcalGainsRcd> ();
    }
    if ((*objectName == "GainWidths") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceGainWidths);
      findingRecord <HcalGainWidthsRcd> ();
    }
    if ((*objectName == "QIEData") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceQIEData);
      findingRecord <HcalQIEDataRcd> ();
    }
    if ((*objectName == "QIETypes") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceQIETypes);
      findingRecord <HcalQIETypesRcd> ();
    }
    if ((*objectName == "ChannelQuality") || (*objectName == "channelQuality") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceChannelQuality);
      findingRecord <HcalChannelQualityRcd> ();
    }
    if ((*objectName == "ElectronicsMap") || (*objectName == "electronicsMap") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceElectronicsMap);
      findingRecord <HcalElectronicsMapRcd> ();
    }
    if ((*objectName == "ZSThresholds") || (*objectName == "zsThresholds") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceZSThresholds);
      findingRecord <HcalZSThresholdsRcd> ();
    }
    if ((*objectName == "RespCorrs") || (*objectName == "ResponseCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceRespCorrs);
      findingRecord <HcalRespCorrsRcd> ();
    }
    if ((*objectName == "LUTCorrs") || (*objectName == "LUTCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceLUTCorrs);
      findingRecord <HcalLUTCorrsRcd> ();
    }
    if ((*objectName == "PFCorrs") || (*objectName == "PFCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::producePFCorrs);
      findingRecord <HcalPFCorrsRcd> ();
    }
    if ((*objectName == "TimeCorrs") || (*objectName == "TimeCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceTimeCorrs);
      findingRecord <HcalTimeCorrsRcd> ();
    }
    if ((*objectName == "L1TriggerObjects") || (*objectName == "L1Trigger") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceL1TriggerObjects);
      findingRecord <HcalL1TriggerObjectsRcd> ();
    }
    if ((*objectName == "ValidationCorrs") || (*objectName == "ValidationCorrection") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceValidationCorrs);
      findingRecord <HcalValidationCorrsRcd> ();
    }
    if ((*objectName == "LutMetadata") || (*objectName == "lutMetadata") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceLutMetadata);
      findingRecord <HcalLutMetadataRcd> ();
    }
    if ((*objectName == "DcsValues") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceDcsValues);
      findingRecord <HcalDcsRcd> ();
    }
    if ((*objectName == "DcsMap") || (*objectName == "dcsMap") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceDcsMap);
      findingRecord <HcalDcsMapRcd> ();
    }
    if ((*objectName == "RecoParams") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceRecoParams);
      findingRecord <HcalRecoParamsRcd> ();
    }
    if ((*objectName == "LongRecoParams") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceLongRecoParams);
      findingRecord <HcalLongRecoParamsRcd> ();
    }
    if ((*objectName == "ZDCLowGainFractions") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceZDCLowGainFractions);
      findingRecord <HcalZDCLowGainFractionsRcd> ();
    }
    if ((*objectName == "MCParams") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceMCParams);
      findingRecord <HcalMCParamsRcd> ();
    }
    if ((*objectName == "FlagHFDigiTimeParams") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceFlagHFDigiTimeParams);
      findingRecord <HcalFlagHFDigiTimeParamsRcd> ();
    }
    if ((*objectName == "FrontEndMap") || (*objectName == "frontEndMap") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceFrontEndMap);
      findingRecord <HcalFrontEndMapRcd> ();
    }
    if ((*objectName == "SiPMParameters") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceSiPMParameters);
      findingRecord <HcalSiPMParametersRcd> ();
    }
    if ((*objectName == "SiPMCharacteristics") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceSiPMCharacteristics);
      findingRecord <HcalSiPMCharacteristicsRcd> ();
    }
    if ((*objectName == "TPChannelParameters") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceTPChannelParameters);
      findingRecord <HcalTPChannelParametersRcd> ();
    }
    if ((*objectName == "TPParameters") || all) {
      setWhatProduced (this, &HcalHardcodeCalibrations::produceTPParameters);
      findingRecord <HcalTPParametersRcd> ();
    }
  }
}


HcalHardcodeCalibrations::~HcalHardcodeCalibrations()
{
  if (he_recalibration != 0 ) delete he_recalibration;
  if (hf_recalibration != 0 ) delete hf_recalibration;
}

//
// member functions
//
void 
HcalHardcodeCalibrations::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  std::string record = iKey.name ();
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::setIntervalFor-> key: " << record << " time: " << iTime.eventID() << '/' << iTime.time ().value ();
  oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

std::unique_ptr<HcalPedestals> HcalHardcodeCalibrations::producePedestals (const HcalPedestalsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePedestals-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalPedestals>(topo,false);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalPedestal item = dbHardcode.makePedestal (*cell, false);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalPedestalWidths> HcalHardcodeCalibrations::producePedestalWidths (const HcalPedestalWidthsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePedestalWidths-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalPedestalWidths>(topo,false);
  std::vector <HcalGenericDetId> cells = allCells(*htopo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalPedestalWidth item = dbHardcode.makePedestalWidth (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalGains> HcalHardcodeCalibrations::produceGains (const HcalGainsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceGains-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalGains>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalGain item = dbHardcode.makeGain (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalGainWidths> HcalHardcodeCalibrations::produceGainWidths (const HcalGainWidthsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceGainWidths-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalGainWidths>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {

    // for Upgrade - include TrigPrims, for regular case - only HcalDetId 
    if(switchGainWidthsForTrigPrims) {
      HcalGainWidth item = dbHardcode.makeGainWidth (*cell);
      result->addValues(item);
    } else if (!cell->isHcalTrigTowerDetId()) {
      HcalGainWidth item = dbHardcode.makeGainWidth (*cell);
      result->addValues(item);
    }
  }
  return result;
}

std::unique_ptr<HcalQIEData> HcalHardcodeCalibrations::produceQIEData (const HcalQIEDataRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceQIEData-> ...";

  /*
  std::cout << std::endl << ">>>  HcalHardcodeCalibrations::produceQIEData"
	    << std::endl;  
  */

  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalQIEData>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalQIECoder coder = dbHardcode.makeQIECoder (*cell);
    result->addCoder (coder);
  }
  return result;
}

std::unique_ptr<HcalQIETypes> HcalHardcodeCalibrations::produceQIETypes (const HcalQIETypesRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceQIETypes-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

    auto result = std::make_unique<HcalQIETypes>(topo);
    std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalQIEType item = dbHardcode.makeQIEType(*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalChannelQuality> HcalHardcodeCalibrations::produceChannelQuality (const HcalChannelQualityRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceChannelQuality-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalChannelQuality>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalChannelStatus item(cell->rawId(),0);
    result->addValues(item);
  }
  return result;
}


std::unique_ptr<HcalRespCorrs> HcalHardcodeCalibrations::produceRespCorrs (const HcalRespCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceRespCorrs-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
 
  //set depth segmentation for HB/HE recalib - only happens once
//  if((he_recalibration && !setHEdsegm) || (hb_recalibration && !setHBdsegm)){
  if((he_recalibration && !setHEdsegm)) {
    std::vector<std::vector<int>> m_segmentation;
    int maxEta = topo->lastHERing();
    m_segmentation.resize(maxEta);
    for (int i = 0; i < maxEta; i++) {
      topo->getDepthSegmentation(i+1,m_segmentation[i]);
    }
    if(he_recalibration && !setHEdsegm){
      he_recalibration->setDsegm(m_segmentation);
      setHEdsegm = true;
    }
    /*
    if(hb_recalibration && !setHBdsegm){
      hb_recalibration->setDsegm(m_segmentation);
      setHBdsegm = true;
    }
    */
  }
 
  auto result = std::make_unique<HcalRespCorrs>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (const auto& cell : cells) {

    double corr = 1.0; 

    //check for layer 0 reweighting: when depth 1 has only one layer, it is layer 0
    if( useLayer0Weight && 
      ((cell.genericSubdet() == HcalGenericDetId::HcalGenEndcap) || (cell.genericSubdet() == HcalGenericDetId::HcalGenBarrel)) &&
      (HcalDetId(cell).depth()==1 && dbHardcode.getLayersInDepth(HcalDetId(cell).ietaAbs(),HcalDetId(cell).depth(),topo)==1) )
    {
      //layer 0 is thicker than other layers (9mm vs 3.7mm) and brighter (Bicron vs SCSN81)
      //in Run1/Run2 (pre-2017 for HE), ODU for layer 0 had neutral density filter attached
      //NDF was simulated as weight of 0.5 applied to Geant energy deposits
      //for Phase1, NDF is removed - simulated as weight of 1.2 applied to Geant energy deposits
      //to maintain RECO calibrations, move the layer 0 energy scale back to its previous state using respcorrs
      corr = 0.5/1.2;
    }

    if ((he_recalibration != 0 ) && (cell.genericSubdet() == HcalGenericDetId::HcalGenEndcap)) {
      int depth_ = HcalDetId(cell).depth();
      int ieta_  = HcalDetId(cell).ieta();
      corr *= he_recalibration->getCorr(ieta_, depth_); 
#ifdef DebugLog      
      std::cout << "HE ieta, depth = " << ieta_  << ",  " << depth_ << "   corr = "  << corr << std::endl;
#endif
    }
    else if ((hf_recalibration != 0 ) && (cell.genericSubdet() == HcalGenericDetId::HcalGenForward)) {
      int depth_ = HcalDetId(cell).depth();
      int ieta_  = HcalDetId(cell).ieta();
      corr = hf_recalibration->getCorr(ieta_, depth_, iLumi); 
#ifdef DebugLog
      std::cout << "HF ieta, depth = " << ieta_  << ",  " << depth_ << "   corr = "  << corr << std::endl;
#endif
    }

    HcalRespCorr item(cell.rawId(),corr);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalLUTCorrs> HcalHardcodeCalibrations::produceLUTCorrs (const HcalLUTCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLUTCorrs-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalLUTCorrs>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalLUTCorr item(cell->rawId(),1.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalPFCorrs> HcalHardcodeCalibrations::producePFCorrs (const HcalPFCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePFCorrs-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalPFCorrs>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalPFCorr item(cell->rawId(),1.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalTimeCorrs> HcalHardcodeCalibrations::produceTimeCorrs (const HcalTimeCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTimeCorrs-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalTimeCorrs>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalTimeCorr item(cell->rawId(),0.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalZSThresholds> HcalHardcodeCalibrations::produceZSThresholds (const HcalZSThresholdsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceZSThresholds-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalZSThresholds>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalZSThreshold item(cell->rawId(),0);
    result->addValues(item);
  }
  return result;
}


std::unique_ptr<HcalL1TriggerObjects> HcalHardcodeCalibrations::produceL1TriggerObjects (const HcalL1TriggerObjectsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceL1TriggerObjects-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalL1TriggerObjects>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalL1TriggerObject item(cell->rawId(),0., 1., 0);
    result->addValues(item);
  }
  // add tag and algo values
  result->setTagString("hardcoded");
  result->setAlgoString("hardcoded");
  return result;
}


std::unique_ptr<HcalElectronicsMap> HcalHardcodeCalibrations::produceElectronicsMap (const HcalElectronicsMapRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceElectronicsMap-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  auto result = std::make_unique<HcalElectronicsMap>();
  dbHardcode.makeHardcodeMap(*result,cells);
  return result;
}

std::unique_ptr<HcalValidationCorrs> HcalHardcodeCalibrations::produceValidationCorrs (const HcalValidationCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceValidationCorrs-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalValidationCorrs>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalValidationCorr item(cell->rawId(),1.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalLutMetadata> HcalHardcodeCalibrations::produceLutMetadata (const HcalLutMetadataRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLutMetadata-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rcd.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalLutMetadata>(topo);

  result->setRctLsb( 0.5 );
  result->setNominalGain(0.177);  // for HBHE SiPMs

  std::vector <HcalGenericDetId> cells = allCells(*topo,dbHardcode.killHE());
  for (const auto& cell: cells) {
    float rcalib = 1.;
    int granularity = 1;
    int threshold = 1;

    if (dbHardcode.useHEUpgrade() or dbHardcode.useHFUpgrade()) {
       // Use values from 2016 as starting conditions for 2017+.  These are
       // averaged over the subdetectors, with the last two HE towers split
       // off due to diverging correction values.
       switch (cell.genericSubdet()) {
          case HcalGenericDetId::HcalGenBarrel:
             rcalib = 1.128;
             break;
         case HcalGenericDetId::HcalGenEndcap:
             {
                HcalDetId id(cell);
                if (id.ietaAbs() >= 28)
                   rcalib = 1.188;
                else
                   rcalib = 1.117;
             }
             break;
         case HcalGenericDetId::HcalGenForward:
             rcalib = 1.02;
             break;
         default:
             break;
       }

       if (cell.isHcalTrigTowerDetId()) {
          rcalib = 0.;
       }
    }

    HcalLutMetadatum item(cell.rawId(), rcalib, granularity, threshold);
    result->addValues(item);
  }
  
  return result;
}

std::unique_ptr<HcalDcsValues> HcalHardcodeCalibrations::produceDcsValues (const HcalDcsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceDcsValues-> ...";
  auto result = std::make_unique<HcalDcsValues>();
  return result;
}

std::unique_ptr<HcalDcsMap> HcalHardcodeCalibrations::produceDcsMap (const HcalDcsMapRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceDcsMap-> ...";

  auto result = std::make_unique<HcalDcsMap>();
  dbHardcode.makeHardcodeDcsMap(*result);
  return result;
}

std::unique_ptr<HcalRecoParams> HcalHardcodeCalibrations::produceRecoParams (const HcalRecoParamsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceRecoParams-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalRecoParams>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalRecoParam item = dbHardcode.makeRecoParam (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalTimingParams> HcalHardcodeCalibrations::produceTimingParams (const HcalTimingParamsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTimingParams-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalTimingParams>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalTimingParam item = dbHardcode.makeTimingParam (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalLongRecoParams> HcalHardcodeCalibrations::produceLongRecoParams (const HcalLongRecoParamsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLongRecoParams-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalLongRecoParams>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  std::vector <unsigned int> mSignal; 
  mSignal.push_back(4); 
  mSignal.push_back(5); 
  mSignal.push_back(6);
  std::vector <unsigned int> mNoise;  
  mNoise.push_back(1);  
  mNoise.push_back(2);  
  mNoise.push_back(3);
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    if (cell->isHcalZDCDetId())
      {
	HcalLongRecoParam item(cell->rawId(),mSignal,mNoise);
	result->addValues(item);
      }
  }
  return result;
}

std::unique_ptr<HcalZDCLowGainFractions> HcalHardcodeCalibrations::produceZDCLowGainFractions (const HcalZDCLowGainFractionsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceZDCLowGainFractions-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalZDCLowGainFractions>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalZDCLowGainFraction item(cell->rawId(),0.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalMCParams> HcalHardcodeCalibrations::produceMCParams (const HcalMCParamsRcd& rec) {


  //  std::cout << std::endl << " .... HcalHardcodeCalibrations::produceMCParams ->"<< std::endl;

  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceMCParams-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  auto result = std::make_unique<HcalMCParams>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {

    //    HcalMCParam item(cell->rawId(),0);
    HcalMCParam item = dbHardcode.makeMCParam (*cell);
    result->addValues(item);
  }
  return result;
}


std::unique_ptr<HcalFlagHFDigiTimeParams> HcalHardcodeCalibrations::produceFlagHFDigiTimeParams (const HcalFlagHFDigiTimeParamsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceFlagHFDigiTimeParams-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalFlagHFDigiTimeParams>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());
  
  std::vector<double> coef;
  coef.push_back(0.93);
  coef.push_back(-0.38275);
  coef.push_back(-0.012667);

  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalFlagHFDigiTimeParam item(cell->rawId(),
				 1, //firstsample
				 3, // samplestoadd
				 2, //expectedpeak
				 40., // min energy threshold
				 coef // coefficients
				 );
    result->addValues(item);
  }
  return result;
} 


std::unique_ptr<HcalFrontEndMap> HcalHardcodeCalibrations::produceFrontEndMap (const HcalFrontEndMapRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceFrontEndMap-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  std::vector <HcalGenericDetId> cells = allCells(*topo, dbHardcode.killHE());

  auto result = std::make_unique<HcalFrontEndMap>();
  dbHardcode.makeHardcodeFrontEndMap(*result, cells);
  return result;
}


std::unique_ptr<HcalSiPMParameters> HcalHardcodeCalibrations::produceSiPMParameters (const HcalSiPMParametersRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceSiPMParameters-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalSiPMParameters>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*htopo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalSiPMParameter item = dbHardcode.makeHardcodeSiPMParameter (*cell,topo);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalSiPMCharacteristics> HcalHardcodeCalibrations::produceSiPMCharacteristics (const HcalSiPMCharacteristicsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceSiPMCharacteristics-> ...";

  auto result = std::make_unique<HcalSiPMCharacteristics>();
  dbHardcode.makeHardcodeSiPMCharacteristics(*result);
  return result;
}


std::unique_ptr<HcalTPChannelParameters> HcalHardcodeCalibrations::produceTPChannelParameters (const HcalTPChannelParametersRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTPChannelParameters-> ...";
  edm::ESHandle<HcalTopology> htopo;
  rec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  auto result = std::make_unique<HcalTPChannelParameters>(topo);
  std::vector <HcalGenericDetId> cells = allCells(*htopo, dbHardcode.killHE());
  for (std::vector <HcalGenericDetId>::const_iterator cell = cells.begin (); cell != cells.end (); ++cell) {
    HcalTPChannelParameter item = dbHardcode.makeHardcodeTPChannelParameter (*cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalTPParameters> HcalHardcodeCalibrations::produceTPParameters (const HcalTPParametersRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTPParameters-> ...";

  auto result = std::make_unique<HcalTPParameters>();
  dbHardcode.makeHardcodeTPParameters(*result);
  return result;
}

void HcalHardcodeCalibrations::fillDescriptions(edm::ConfigurationDescriptions & descriptions){
	edm::ParameterSetDescription desc;
	desc.add<double>("iLumi",-1.);
	desc.add<bool>("HERecalibration",false);
	desc.add<double>("HEreCalibCutoff",20.);
	desc.add<bool>("HFRecalibration",false);
	desc.add<bool>("GainWidthsForTrigPrims",false);
	desc.add<bool>("useHBUpgrade",false);
	desc.add<bool>("useHEUpgrade",false);
	desc.add<bool>("useHFUpgrade",false);
	desc.add<bool>("useHOUpgrade",true);
	desc.add<bool>("testHFQIE10",false);
	desc.add<bool>("testHEPlan1",false);
	desc.add<bool>("killHE",false);
	desc.add<bool>("useLayer0Weight",false);
	desc.addUntracked<std::vector<std::string> >("toGet",std::vector<std::string>());
	desc.addUntracked<bool>("fromDDD",false);
	
	edm::ParameterSetDescription desc_hb;
	desc_hb.add<std::vector<double>>("gain", std::vector<double>({0.19}));
	desc_hb.add<std::vector<double>>("gainWidth", std::vector<double>({0.0}));
	desc_hb.add<double>("pedestal", 3.0);
	desc_hb.add<double>("pedestalWidth", 0.55);
	desc_hb.add<std::vector<double>>("qieOffset", std::vector<double>({-0.49, 1.8, 7.2, 37.9}));
	desc_hb.add<std::vector<double>>("qieSlope", std::vector<double>({0.912, 0.917, 0.922, 0.923}));
	desc_hb.add<int>("qieType", 0);
	desc_hb.add<int>("mcShape",125);
	desc_hb.add<int>("recoShape",105);
	desc_hb.add<double>("photoelectronsToAnalog",0.0);
	desc_hb.add<std::vector<double>>("darkCurrent",std::vector<double>(0.0));
	desc.add<edm::ParameterSetDescription>("hb", desc_hb);

	edm::ParameterSetDescription desc_hbUpgrade;
	desc_hbUpgrade.add<std::vector<double>>("gain", std::vector<double>({0.00111111111111}));
	desc_hbUpgrade.add<std::vector<double>>("gainWidth", std::vector<double>({0}));
	desc_hbUpgrade.add<double>("pedestal", 18.0);
	desc_hbUpgrade.add<double>("pedestalWidth", 5.0);
	desc_hbUpgrade.add<std::vector<double>>("qieOffset", std::vector<double>({0.0, 0.0, 0.0, 0.0}));
	desc_hbUpgrade.add<std::vector<double>>("qieSlope", std::vector<double>({0.333, 0.333, 0.333, 0.333}));
	desc_hbUpgrade.add<int>("qieType", 2);
	desc_hbUpgrade.add<int>("mcShape",203);
	desc_hbUpgrade.add<int>("recoShape",203);
	desc_hbUpgrade.add<double>("photoelectronsToAnalog",57.5);
	desc_hbUpgrade.add<std::vector<double>>("darkCurrent",std::vector<double>(0.055));
	desc.add<edm::ParameterSetDescription>("hbUpgrade", desc_hbUpgrade);

	edm::ParameterSetDescription desc_he;
	desc_he.add<std::vector<double>>("gain", std::vector<double>({0.23}));
	desc_he.add<std::vector<double>>("gainWidth", std::vector<double>({0}));
	desc_he.add<double>("pedestal", 3.0);
	desc_he.add<double>("pedestalWidth", 0.79);
	desc_he.add<std::vector<double>>("qieOffset", std::vector<double>({-0.38, 2.0, 7.6, 39.6}));
	desc_he.add<std::vector<double>>("qieSlope", std::vector<double>({0.912, 0.916, 0.92, 0.922}));
	desc_he.add<int>("qieType", 0);
	desc_he.add<int>("mcShape",125);
	desc_he.add<int>("recoShape",105);
	desc_he.add<double>("photoelectronsToAnalog",0.0);
	desc_he.add<std::vector<double>>("darkCurrent",std::vector<double>(0.0));
	desc.add<edm::ParameterSetDescription>("he", desc_he);

	edm::ParameterSetDescription desc_heUpgrade;
	desc_heUpgrade.add<std::vector<double>>("gain", std::vector<double>({0.00111111111111}));
	desc_heUpgrade.add<std::vector<double>>("gainWidth", std::vector<double>({0}));
	desc_heUpgrade.add<double>("pedestal", 18.0);
	desc_heUpgrade.add<double>("pedestalWidth", 5.0);
	desc_heUpgrade.add<std::vector<double>>("qieOffset", std::vector<double>({0.0, 0.0, 0.0, 0.0}));
	desc_heUpgrade.add<std::vector<double>>("qieSlope", std::vector<double>({0.333, 0.333, 0.333, 0.333}));
	desc_heUpgrade.add<int>("qieType", 2);
	desc_heUpgrade.add<int>("mcShape",203);
	desc_heUpgrade.add<int>("recoShape",203);
	desc_heUpgrade.add<double>("photoelectronsToAnalog",57.5);
	desc_heUpgrade.add<std::vector<double>>("darkCurrent",std::vector<double>(0.055));
	desc.add<edm::ParameterSetDescription>("heUpgrade", desc_heUpgrade);

	edm::ParameterSetDescription desc_hf;
	desc_hf.add<std::vector<double>>("gain", std::vector<double>({0.14, 0.135}));
	desc_hf.add<std::vector<double>>("gainWidth", std::vector<double>({0.0, 0.0}));
	desc_hf.add<double>("pedestal", 3.0);
	desc_hf.add<double>("pedestalWidth", 0.84);
	desc_hf.add<std::vector<double>>("qieOffset", std::vector<double>({-0.87, 1.4, 7.8, -29.6}));
	desc_hf.add<std::vector<double>>("qieSlope", std::vector<double>({0.359, 0.358, 0.36, 0.367}));
	desc_hf.add<int>("qieType", 0);
	desc_hf.add<int>("mcShape",301);
	desc_hf.add<int>("recoShape",301);
	desc_hf.add<double>("photoelectronsToAnalog",0.0);
	desc_hf.add<std::vector<double>>("darkCurrent",std::vector<double>(0.0));
	desc.add<edm::ParameterSetDescription>("hf", desc_hf);

	edm::ParameterSetDescription desc_hfUpgrade;
	desc_hfUpgrade.add<std::vector<double>>("gain", std::vector<double>({0.14, 0.135}));
	desc_hfUpgrade.add<std::vector<double>>("gainWidth", std::vector<double>({0.0, 0.0}));
	desc_hfUpgrade.add<double>("pedestal", 13.33);
	desc_hfUpgrade.add<double>("pedestalWidth", 3.33);
	desc_hfUpgrade.add<std::vector<double>>("qieOffset", std::vector<double>({0.0697, -0.7405, 12.38, -671.9}));
	desc_hfUpgrade.add<std::vector<double>>("qieSlope", std::vector<double>({0.297, 0.298, 0.298, 0.313}));
	desc_hfUpgrade.add<int>("qieType", 1);
	desc_hfUpgrade.add<int>("mcShape",301);
	desc_hfUpgrade.add<int>("recoShape",301);
	desc_hfUpgrade.add<double>("photoelectronsToAnalog",0.0);
	desc_hfUpgrade.add<std::vector<double>>("darkCurrent",std::vector<double>(0.0));
	desc.add<edm::ParameterSetDescription>("hfUpgrade", desc_hfUpgrade);
  
	edm::ParameterSetDescription desc_hfrecal;
	desc_hfrecal.add<std::vector<double>>("HFdepthOneParameterA", std::vector<double>());
	desc_hfrecal.add<std::vector<double>>("HFdepthOneParameterB", std::vector<double>());
	desc_hfrecal.add<std::vector<double>>("HFdepthTwoParameterA", std::vector<double>());
	desc_hfrecal.add<std::vector<double>>("HFdepthTwoParameterB", std::vector<double>());
	desc.add<edm::ParameterSetDescription>("HFRecalParameterBlock", desc_hfrecal);

	edm::ParameterSetDescription desc_ho;
	desc_ho.add<std::vector<double>>("gain", std::vector<double>({0.006, 0.0087}));
	desc_ho.add<std::vector<double>>("gainWidth", std::vector<double>({0.0, 0.0}));
	desc_ho.add<double>("pedestal", 11.0);
	desc_ho.add<double>("pedestalWidth", 0.57);
	desc_ho.add<std::vector<double>>("qieOffset", std::vector<double>({-0.44, 1.4, 7.1, 38.5}));
	desc_ho.add<std::vector<double>>("qieSlope", std::vector<double>({0.907, 0.915, 0.92, 0.921}));
	desc_ho.add<int>("qieType", 0);
	desc_ho.add<int>("mcShape",201);
	desc_ho.add<int>("recoShape",201);
	desc_ho.add<double>("photoelectronsToAnalog",4.0);
	desc_ho.add<std::vector<double>>("darkCurrent",std::vector<double>(0.0));
	desc.add<edm::ParameterSetDescription>("ho", desc_ho);

	edm::ParameterSetDescription validator_sipm;
	validator_sipm.add<int>("pixels",1);
	validator_sipm.add<double>("crosstalk",0);
	validator_sipm.add<double>("nonlin1",1);
	validator_sipm.add<double>("nonlin2",0);
	validator_sipm.add<double>("nonlin3",0);
	std::vector<edm::ParameterSet> default_sipm(1);
	desc.addVPSet("SiPMCharacteristics",validator_sipm,default_sipm);
	
	descriptions.addDefault(desc);
}
