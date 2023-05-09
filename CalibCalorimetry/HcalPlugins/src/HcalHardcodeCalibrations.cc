// -*- C++ -*-
// Original Author:  Fedor Ratnikov
//
//

#include <memory>
#include <iostream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/ValidityInterval.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

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

  const std::vector<HcalGenericDetId>& allCells(const HcalTopology& hcaltopology, bool killHE = false) {
    static std::vector<HcalGenericDetId> result;
    int maxDepth = hcaltopology.maxDepth();

#ifdef DebugLog
    std::cout << std::endl << "HcalHardcodeCalibrations:   maxDepth = " << maxDepth << std::endl;
#endif

    if (result.empty()) {
      for (int eta = -HcalDetId::kHcalEtaMask2; eta <= (int)(HcalDetId::kHcalEtaMask2); eta++) {
        for (unsigned int phi = 0; phi <= HcalDetId::kHcalPhiMask2; phi++) {
          for (int depth = 1; depth <= maxDepth; depth++) {
            for (int det = 1; det <= HcalForward; det++) {
              HcalDetId cell((HcalSubdetector)det, eta, phi, depth);
              if (killHE && HcalEndcap == cell.subdetId())
                continue;
              if (hcaltopology.valid(cell)) {
                result.push_back(cell);
#ifdef DebugLog
                std::cout << " HcalHardcodedCalibrations: det|eta|phi|depth = " << det << "|" << eta << "|" << phi
                          << "|" << depth << std::endl;
#endif
              }
            }
          }
        }
      }
      ZdcTopology zdctopology;
      HcalZDCDetId zcell;
      HcalZDCDetId::Section section = HcalZDCDetId::EM;
      for (int depth = 1; depth < 6; depth++) {
        zcell = HcalZDCDetId(section, true, depth);
        if (zdctopology.valid(zcell))
          result.push_back(zcell);
        zcell = HcalZDCDetId(section, false, depth);
        if (zdctopology.valid(zcell))
          result.push_back(zcell);
      }
      section = HcalZDCDetId::HAD;
      for (int depth = 1; depth < 5; depth++) {
        zcell = HcalZDCDetId(section, true, depth);
        if (zdctopology.valid(zcell))
          result.push_back(zcell);
        zcell = HcalZDCDetId(section, false, depth);
        if (zdctopology.valid(zcell))
          result.push_back(zcell);
      }
      section = HcalZDCDetId::LUM;
      for (int depth = 1; depth < 3; depth++) {
        zcell = HcalZDCDetId(section, true, depth);
        if (zdctopology.valid(zcell))
          result.push_back(zcell);
        zcell = HcalZDCDetId(section, false, depth);
        if (zdctopology.valid(zcell))
          result.push_back(zcell);
      }
      section = HcalZDCDetId::RPD;
      for (int depth = 1; depth < 17; depth++) {
        zcell = HcalZDCDetId(section, true, depth);
        if (zdctopology.valid(zcell))
          result.push_back(zcell);
        zcell = HcalZDCDetId(section, false, depth);
        if (zdctopology.valid(zcell))
          result.push_back(zcell);
      }

      // HcalGenTriggerTower (HcalGenericSubdetector = 5)
      // NASTY HACK !!!
      // - As no valid(cell) check found for HcalTrigTowerDetId
      // to create HT cells (ieta=1-28, iphi=1-72)&(ieta=29-32, iphi=1,5,... 69)

      for (int vers = 0; vers <= HcalTrigTowerDetId::kHcalVersMask; ++vers) {
        for (int depth = 0; depth <= HcalTrigTowerDetId::kHcalDepthMask; ++depth) {
          for (int eta = -HcalTrigTowerDetId::kHcalEtaMask; eta <= HcalTrigTowerDetId::kHcalEtaMask; eta++) {
            for (int phi = 1; phi <= HcalTrigTowerDetId::kHcalPhiMask; phi++) {
              HcalTrigTowerDetId cell(eta, phi, depth, vers);
              if (hcaltopology.validHT(cell)) {
                result.push_back(cell);
#ifdef DebugLog
                std::cout << " HcalHardcodedCalibrations: eta|phi|depth|vers = " << eta << "|" << phi << "|" << depth
                          << "|" << vers << std::endl;
#endif
              }
            }
          }
        }
      }
    }
    return result;
  }

}  // namespace

HcalHardcodeCalibrations::HcalHardcodeCalibrations(const edm::ParameterSet& iConfig)
    : hb_recalibration(nullptr),
      he_recalibration(nullptr),
      hf_recalibration(nullptr),
      setHEdsegm(false),
      setHBdsegm(false) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::HcalHardcodeCalibrations->...";

  if (iConfig.exists("GainWidthsForTrigPrims"))
    switchGainWidthsForTrigPrims = iConfig.getParameter<bool>("GainWidthsForTrigPrims");
  else
    switchGainWidthsForTrigPrims = false;

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
  useIeta18depth1 = iConfig.getParameter<bool>("useIeta18depth1");
  testHEPlan1 = iConfig.getParameter<bool>("testHEPlan1");
  // HB, HE, HF recalibration preparation
  iLumi = iConfig.getParameter<double>("iLumi");

  if (iLumi > 0.0) {
    bool hb_recalib = iConfig.getParameter<bool>("HBRecalibration");
    bool he_recalib = iConfig.getParameter<bool>("HERecalibration");
    bool hf_recalib = iConfig.getParameter<bool>("HFRecalibration");
    if (hb_recalib) {
      hb_recalibration =
          std::make_unique<HBHERecalibration>(iLumi,
                                              iConfig.getParameter<double>("HBreCalibCutoff"),
                                              iConfig.getParameter<edm::FileInPath>("HBmeanenergies").fullPath());
    }
    if (he_recalib) {
      he_recalibration =
          std::make_unique<HBHERecalibration>(iLumi,
                                              iConfig.getParameter<double>("HEreCalibCutoff"),
                                              iConfig.getParameter<edm::FileInPath>("HEmeanenergies").fullPath());
    }
    if (hf_recalib && !iConfig.getParameter<edm::ParameterSet>("HFRecalParameterBlock").empty())
      hf_recalibration =
          std::make_unique<HFRecalibration>(iConfig.getParameter<edm::ParameterSet>("HFRecalParameterBlock"));

#ifdef DebugLog
    std::cout << " HcalHardcodeCalibrations:  iLumi = " << iLumi << std::endl;
#endif
  }

  std::vector<std::string> toGet = iConfig.getUntrackedParameter<std::vector<std::string>>("toGet");
  for (auto& objectName : toGet) {
    bool all = objectName == "all";
#ifdef DebugLog
    std::cout << "Load parameters for " << objectName << std::endl;
#endif
    if ((objectName == "Pedestals") || all) {
      topoTokens_[kPedestals] = setWhatProduced(this, &HcalHardcodeCalibrations::producePedestals).consumes();
      findingRecord<HcalPedestalsRcd>();
    }
    if ((objectName == "PedestalWidths") || all) {
      topoTokens_[kPedestalWidths] = setWhatProduced(this, &HcalHardcodeCalibrations::producePedestalWidths).consumes();
      findingRecord<HcalPedestalWidthsRcd>();
    }
    if ((objectName == "EffectivePedestals") || all) {
      topoTokens_[kEffectivePedestals] =
          setWhatProduced(this, &HcalHardcodeCalibrations::produceEffectivePedestals, edm::es::Label("effective"))
              .consumes();
      findingRecord<HcalPedestalsRcd>();
    }
    if ((objectName == "EffectivePedestalWidths") || all) {
      topoTokens_[kEffectivePedestalWidths] =
          setWhatProduced(this, &HcalHardcodeCalibrations::produceEffectivePedestalWidths, edm::es::Label("effective"))
              .consumes();
      findingRecord<HcalPedestalWidthsRcd>();
    }
    if ((objectName == "Gains") || all) {
      topoTokens_[kGains] = setWhatProduced(this, &HcalHardcodeCalibrations::produceGains).consumes();
      findingRecord<HcalGainsRcd>();
    }
    if ((objectName == "GainWidths") || all) {
      topoTokens_[kGainWidths] = setWhatProduced(this, &HcalHardcodeCalibrations::produceGainWidths).consumes();
      findingRecord<HcalGainWidthsRcd>();
    }
    if ((objectName == "PFCuts") || all) {
      topoTokens_[kPFCuts] = setWhatProduced(this, &HcalHardcodeCalibrations::producePFCuts).consumes();
      findingRecord<HcalPFCutsRcd>();
    }
    if ((objectName == "QIEData") || all) {
      topoTokens_[kQIEData] = setWhatProduced(this, &HcalHardcodeCalibrations::produceQIEData).consumes();
      findingRecord<HcalQIEDataRcd>();
    }
    if ((objectName == "QIETypes") || all) {
      topoTokens_[kQIETypes] = setWhatProduced(this, &HcalHardcodeCalibrations::produceQIETypes).consumes();
      findingRecord<HcalQIETypesRcd>();
    }
    if ((objectName == "ChannelQuality") || (objectName == "channelQuality") || all) {
      topoTokens_[kChannelQuality] = setWhatProduced(this, &HcalHardcodeCalibrations::produceChannelQuality).consumes();
      findingRecord<HcalChannelQualityRcd>();
    }
    if ((objectName == "ElectronicsMap") || (objectName == "electronicsMap") || all) {
      topoTokens_[kElectronicsMap] = setWhatProduced(this, &HcalHardcodeCalibrations::produceElectronicsMap).consumes();
      findingRecord<HcalElectronicsMapRcd>();
    }
    if ((objectName == "ZSThresholds") || (objectName == "zsThresholds") || all) {
      topoTokens_[kZSThresholds] = setWhatProduced(this, &HcalHardcodeCalibrations::produceZSThresholds).consumes();
      findingRecord<HcalZSThresholdsRcd>();
    }
    if ((objectName == "RespCorrs") || (objectName == "ResponseCorrection") || all) {
      auto c = setWhatProduced(this, &HcalHardcodeCalibrations::produceRespCorrs);
      topoTokens_[kRespCorrs] = c.consumes();
      if (he_recalibration) {
        heDarkeningToken_ = c.consumes(edm::ESInputTag("", "HE"));
      }
      if (hb_recalibration) {
        hbDarkeningToken_ = c.consumes(edm::ESInputTag("", "HB"));
      }
      findingRecord<HcalRespCorrsRcd>();
    }
    if ((objectName == "LUTCorrs") || (objectName == "LUTCorrection") || all) {
      topoTokens_[kLUTCorrs] = setWhatProduced(this, &HcalHardcodeCalibrations::produceLUTCorrs).consumes();
      findingRecord<HcalLUTCorrsRcd>();
    }
    if ((objectName == "PFCorrs") || (objectName == "PFCorrection") || all) {
      topoTokens_[kPFCorrs] = setWhatProduced(this, &HcalHardcodeCalibrations::producePFCorrs).consumes();
      findingRecord<HcalPFCorrsRcd>();
    }
    if ((objectName == "TimeCorrs") || (objectName == "TimeCorrection") || all) {
      topoTokens_[kTimeCorrs] = setWhatProduced(this, &HcalHardcodeCalibrations::produceTimeCorrs).consumes();
      findingRecord<HcalTimeCorrsRcd>();
    }
    if ((objectName == "L1TriggerObjects") || (objectName == "L1Trigger") || all) {
      topoTokens_[kL1TriggerObjects] =
          setWhatProduced(this, &HcalHardcodeCalibrations::produceL1TriggerObjects).consumes();
      findingRecord<HcalL1TriggerObjectsRcd>();
    }
    if ((objectName == "ValidationCorrs") || (objectName == "ValidationCorrection") || all) {
      topoTokens_[kValidationCorrs] =
          setWhatProduced(this, &HcalHardcodeCalibrations::produceValidationCorrs).consumes();
      findingRecord<HcalValidationCorrsRcd>();
    }
    if ((objectName == "LutMetadata") || (objectName == "lutMetadata") || all) {
      topoTokens_[kLutMetadata] = setWhatProduced(this, &HcalHardcodeCalibrations::produceLutMetadata).consumes();
      findingRecord<HcalLutMetadataRcd>();
    }
    if ((objectName == "DcsValues") || all) {
      setWhatProduced(this, &HcalHardcodeCalibrations::produceDcsValues);
      findingRecord<HcalDcsRcd>();
    }
    if ((objectName == "DcsMap") || (objectName == "dcsMap") || all) {
      setWhatProduced(this, &HcalHardcodeCalibrations::produceDcsMap);
      findingRecord<HcalDcsMapRcd>();
    }
    if ((objectName == "RecoParams") || all) {
      topoTokens_[kRecoParams] = setWhatProduced(this, &HcalHardcodeCalibrations::produceRecoParams).consumes();
      findingRecord<HcalRecoParamsRcd>();
    }
    if ((objectName == "LongRecoParams") || all) {
      topoTokens_[kLongRecoParams] = setWhatProduced(this, &HcalHardcodeCalibrations::produceLongRecoParams).consumes();
      findingRecord<HcalLongRecoParamsRcd>();
    }
    if ((objectName == "ZDCLowGainFractions") || all) {
      topoTokens_[kZDCLowGainFractions] =
          setWhatProduced(this, &HcalHardcodeCalibrations::produceZDCLowGainFractions).consumes();
      findingRecord<HcalZDCLowGainFractionsRcd>();
    }
    if ((objectName == "MCParams") || all) {
      topoTokens_[kMCParams] = setWhatProduced(this, &HcalHardcodeCalibrations::produceMCParams).consumes();
      findingRecord<HcalMCParamsRcd>();
    }
    if ((objectName == "FlagHFDigiTimeParams") || all) {
      topoTokens_[kFlagHFDigiTimeParams] =
          setWhatProduced(this, &HcalHardcodeCalibrations::produceFlagHFDigiTimeParams).consumes();
      findingRecord<HcalFlagHFDigiTimeParamsRcd>();
    }
    if ((objectName == "FrontEndMap") || (objectName == "frontEndMap") || all) {
      topoTokens_[kFrontEndMap] = setWhatProduced(this, &HcalHardcodeCalibrations::produceFrontEndMap).consumes();
      findingRecord<HcalFrontEndMapRcd>();
    }
    if ((objectName == "SiPMParameters") || all) {
      topoTokens_[kSiPMParameters] = setWhatProduced(this, &HcalHardcodeCalibrations::produceSiPMParameters).consumes();
      findingRecord<HcalSiPMParametersRcd>();
    }
    if ((objectName == "SiPMCharacteristics") || all) {
      setWhatProduced(this, &HcalHardcodeCalibrations::produceSiPMCharacteristics);
      findingRecord<HcalSiPMCharacteristicsRcd>();
    }
    if ((objectName == "TPChannelParameters") || all) {
      topoTokens_[kTPChannelParameters] =
          setWhatProduced(this, &HcalHardcodeCalibrations::produceTPChannelParameters).consumes();
      findingRecord<HcalTPChannelParametersRcd>();
    }
    if ((objectName == "TPParameters") || all) {
      setWhatProduced(this, &HcalHardcodeCalibrations::produceTPParameters);
      findingRecord<HcalTPParametersRcd>();
    }
  }
}

HcalHardcodeCalibrations::~HcalHardcodeCalibrations() {}

//
// member functions
//
void HcalHardcodeCalibrations::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                              const edm::IOVSyncValue& iTime,
                                              edm::ValidityInterval& oInterval) {
  std::string record = iKey.name();
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::setIntervalFor-> key: " << record << " time: " << iTime.eventID()
                       << '/' << iTime.time().value();
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());  //infinite
}

std::unique_ptr<HcalPedestals> HcalHardcodeCalibrations::producePedestals_(
    const HcalPedestalsRcd& rec, const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord>& token, bool eff) {
  std::string seff = eff ? "Effective" : "";
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produce" << seff << "Pedestals-> ...";

  auto const& topo = rec.get(token);
  auto result = std::make_unique<HcalPedestals>(&topo, false);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalPedestal item = dbHardcode.makePedestal(cell, false, eff, &topo, iLumi);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalPedestalWidths> HcalHardcodeCalibrations::producePedestalWidths_(
    const HcalPedestalWidthsRcd& rec, const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord>& token, bool eff) {
  std::string seff = eff ? "Effective" : "";
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produce" << seff << "PedestalWidths-> ...";
  auto const& topo = rec.get(token);
  auto result = std::make_unique<HcalPedestalWidths>(&topo, false);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalPedestalWidth item = dbHardcode.makePedestalWidth(cell, eff, &topo, iLumi);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalPedestals> HcalHardcodeCalibrations::producePedestals(const HcalPedestalsRcd& rec) {
  return producePedestals_(rec, topoTokens_[kPedestals], false);
}

std::unique_ptr<HcalPedestals> HcalHardcodeCalibrations::produceEffectivePedestals(const HcalPedestalsRcd& rec) {
  return producePedestals_(rec, topoTokens_[kEffectivePedestals], true);
}

std::unique_ptr<HcalPedestalWidths> HcalHardcodeCalibrations::producePedestalWidths(const HcalPedestalWidthsRcd& rec) {
  return producePedestalWidths_(rec, topoTokens_[kPedestalWidths], false);
}

std::unique_ptr<HcalPedestalWidths> HcalHardcodeCalibrations::produceEffectivePedestalWidths(
    const HcalPedestalWidthsRcd& rec) {
  return producePedestalWidths_(rec, topoTokens_[kEffectivePedestalWidths], true);
}

std::unique_ptr<HcalGains> HcalHardcodeCalibrations::produceGains(const HcalGainsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceGains-> ...";

  auto const& topo = rec.get(topoTokens_[kGains]);
  auto result = std::make_unique<HcalGains>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalGain item = dbHardcode.makeGain(cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalGainWidths> HcalHardcodeCalibrations::produceGainWidths(const HcalGainWidthsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceGainWidths-> ...";

  auto const& topo = rec.get(topoTokens_[kGainWidths]);
  auto result = std::make_unique<HcalGainWidths>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    // for Upgrade - include TrigPrims, for regular case - only HcalDetId
    if (switchGainWidthsForTrigPrims) {
      HcalGainWidth item = dbHardcode.makeGainWidth(cell);
      result->addValues(item);
    } else if (!cell.isHcalTrigTowerDetId()) {
      HcalGainWidth item = dbHardcode.makeGainWidth(cell);
      result->addValues(item);
    }
  }
  return result;
}

std::unique_ptr<HcalPFCuts> HcalHardcodeCalibrations::producePFCuts(const HcalPFCutsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePFCuts-> ...";

  auto const& topo = rec.get(topoTokens_[kPFCuts]);
  auto result = std::make_unique<HcalPFCuts>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    // Use only standard Hcal channels for now, no TrigPrims
    if (!cell.isHcalTrigTowerDetId()) {
      HcalPFCut item = dbHardcode.makePFCut(cell);
      result->addValues(item);
    }
  }
  return result;
}

std::unique_ptr<HcalQIEData> HcalHardcodeCalibrations::produceQIEData(const HcalQIEDataRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceQIEData-> ...";

  /*
  std::cout << std::endl << ">>>  HcalHardcodeCalibrations::produceQIEData"
	    << std::endl;  
  */

  auto const& topo = rcd.get(topoTokens_[kQIEData]);
  auto result = std::make_unique<HcalQIEData>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalQIECoder coder = dbHardcode.makeQIECoder(cell);
    result->addCoder(coder);
  }
  return result;
}

std::unique_ptr<HcalQIETypes> HcalHardcodeCalibrations::produceQIETypes(const HcalQIETypesRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceQIETypes-> ...";
  auto const& topo = rcd.get(topoTokens_[kQIETypes]);

  auto result = std::make_unique<HcalQIETypes>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalQIEType item = dbHardcode.makeQIEType(cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalChannelQuality> HcalHardcodeCalibrations::produceChannelQuality(const HcalChannelQualityRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceChannelQuality-> ...";
  auto const& topo = rcd.get(topoTokens_[kChannelQuality]);

  auto result = std::make_unique<HcalChannelQuality>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    // Special: removal of (non-instrumented) layer "-1"("nose") = depth 1
    // from Upgrade HE, either from
    // (i)  HEP17 sector in 2017 or
    // (ii) the entire HE rin=18 from 2018 through Run 3.
    // May require a revision  by 2021.

    uint32_t status = 0;

    if (!(cell.isHcalZDCDetId())) {
      HcalDetId hid = HcalDetId(cell);
      int iphi = hid.iphi();
      int ieta = hid.ieta();
      int absieta = hid.ietaAbs();
      int depth = hid.depth();

      // specific HEP17 sector (2017 only)
      bool isHEP17 = (iphi >= 63) && (iphi <= 66) && (ieta > 0);
      // |ieta|=18, depth=1
      bool is18d1 = (absieta == 18) && (depth == 1);

      if ((!useIeta18depth1 && is18d1) && ((testHEPlan1 && isHEP17) || (!testHEPlan1))) {
        status = 0x8002;  // dead cell
      }
    }

    HcalChannelStatus item(cell.rawId(), status);
    result->addValues(item);
  }

  return result;
}

std::unique_ptr<HcalRespCorrs> HcalHardcodeCalibrations::produceRespCorrs(const HcalRespCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceRespCorrs-> ...";
  auto const& topo = rcd.get(topoTokens_[kRespCorrs]);

  //set depth segmentation for HB/HE recalib - only happens once
  if ((he_recalibration && !setHEdsegm) || (hb_recalibration && !setHBdsegm)) {
    std::vector<std::vector<int>> m_segmentation;
    int maxEta = topo.lastHBHERing();
    m_segmentation.resize(maxEta);
    for (int i = 0; i < maxEta; i++) {
      topo.getDepthSegmentation(i + 1, m_segmentation[i]);
    }
    if (he_recalibration && !setHEdsegm) {
      he_recalibration->setup(m_segmentation, &rcd.get(heDarkeningToken_));
      setHEdsegm = true;
    }
    if (hb_recalibration && !setHBdsegm) {
      hb_recalibration->setup(m_segmentation, &rcd.get(hbDarkeningToken_));
      setHBdsegm = true;
    }
  }

  auto result = std::make_unique<HcalRespCorrs>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (const auto& cell : cells) {
    double corr = 1.0;

    //check for layer 0 reweighting: when depth 1 has only one layer, it is layer 0
    if (useLayer0Weight &&
        ((cell.genericSubdet() == HcalGenericDetId::HcalGenEndcap) ||
         (cell.genericSubdet() == HcalGenericDetId::HcalGenBarrel)) &&
        (HcalDetId(cell).depth() == 1 &&
         dbHardcode.getLayersInDepth(HcalDetId(cell).ietaAbs(), HcalDetId(cell).depth(), &topo) == 1)) {
      //layer 0 is thicker than other layers (9mm vs 3.7mm) and brighter (Bicron vs SCSN81)
      //in Run1/Run2 (pre-2017 for HE), ODU for layer 0 had neutral density filter attached
      //NDF was simulated as weight of 0.5 applied to Geant energy deposits
      //for Phase1, NDF is removed - simulated as weight of 1.2 applied to Geant energy deposits
      //to maintain RECO calibrations, move the layer 0 energy scale back to its previous state using respcorrs
      corr = 0.5 / 1.2;
    }

    if ((hb_recalibration != nullptr) && (cell.genericSubdet() == HcalGenericDetId::HcalGenBarrel)) {
      int depth_ = HcalDetId(cell).depth();
      int ieta_ = HcalDetId(cell).ieta();
      corr *= hb_recalibration->getCorr(ieta_, depth_);
#ifdef DebugLog
      std::cout << "HB ieta, depth = " << ieta_ << ",  " << depth_ << "   corr = " << corr << std::endl;
#endif
    } else if ((he_recalibration != nullptr) && (cell.genericSubdet() == HcalGenericDetId::HcalGenEndcap)) {
      int depth_ = HcalDetId(cell).depth();
      int ieta_ = HcalDetId(cell).ieta();
      corr *= he_recalibration->getCorr(ieta_, depth_);
#ifdef DebugLog
      std::cout << "HE ieta, depth = " << ieta_ << ",  " << depth_ << "   corr = " << corr << std::endl;
#endif
    } else if ((hf_recalibration != nullptr) && (cell.genericSubdet() == HcalGenericDetId::HcalGenForward)) {
      int depth_ = HcalDetId(cell).depth();
      int ieta_ = HcalDetId(cell).ieta();
      corr = hf_recalibration->getCorr(ieta_, depth_, iLumi);
#ifdef DebugLog
      std::cout << "HF ieta, depth = " << ieta_ << ",  " << depth_ << "   corr = " << corr << std::endl;
#endif
    }

    HcalRespCorr item(cell.rawId(), corr);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalLUTCorrs> HcalHardcodeCalibrations::produceLUTCorrs(const HcalLUTCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLUTCorrs-> ...";
  auto const& topo = rcd.get(topoTokens_[kLUTCorrs]);

  auto result = std::make_unique<HcalLUTCorrs>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalLUTCorr item(cell.rawId(), 1.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalPFCorrs> HcalHardcodeCalibrations::producePFCorrs(const HcalPFCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::producePFCorrs-> ...";
  auto const& topo = rcd.get(topoTokens_[kPFCorrs]);

  auto result = std::make_unique<HcalPFCorrs>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalPFCorr item(cell.rawId(), 1.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalTimeCorrs> HcalHardcodeCalibrations::produceTimeCorrs(const HcalTimeCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTimeCorrs-> ...";
  auto const& topo = rcd.get(topoTokens_[kTimeCorrs]);

  auto result = std::make_unique<HcalTimeCorrs>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalTimeCorr item(cell.rawId(), 0.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalZSThresholds> HcalHardcodeCalibrations::produceZSThresholds(const HcalZSThresholdsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceZSThresholds-> ...";
  auto const& topo = rcd.get(topoTokens_[kZSThresholds]);

  auto result = std::make_unique<HcalZSThresholds>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalZSThreshold item = dbHardcode.makeZSThreshold(cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalL1TriggerObjects> HcalHardcodeCalibrations::produceL1TriggerObjects(
    const HcalL1TriggerObjectsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceL1TriggerObjects-> ...";
  auto const& topo = rcd.get(topoTokens_[kL1TriggerObjects]);

  auto result = std::make_unique<HcalL1TriggerObjects>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalL1TriggerObject item(cell.rawId(), 0., 1., 0);
    result->addValues(item);
  }
  // add tag and algo values
  result->setTagString("hardcoded");
  result->setAlgoString("hardcoded");
  return result;
}

std::unique_ptr<HcalElectronicsMap> HcalHardcodeCalibrations::produceElectronicsMap(const HcalElectronicsMapRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceElectronicsMap-> ...";
  auto const& topo = rcd.get(topoTokens_[kElectronicsMap]);

  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  return dbHardcode.makeHardcodeMap(cells);
}

std::unique_ptr<HcalValidationCorrs> HcalHardcodeCalibrations::produceValidationCorrs(
    const HcalValidationCorrsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceValidationCorrs-> ...";
  auto const& topo = rcd.get(topoTokens_[kValidationCorrs]);

  auto result = std::make_unique<HcalValidationCorrs>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalValidationCorr item(cell.rawId(), 1.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalLutMetadata> HcalHardcodeCalibrations::produceLutMetadata(const HcalLutMetadataRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLutMetadata-> ...";
  auto const& topo = rcd.get(topoTokens_[kLutMetadata]);

  auto result = std::make_unique<HcalLutMetadata>(&topo);

  result->setRctLsb(0.5);
  result->setNominalGain(0.177);  // for HBHE SiPMs

  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (const auto& cell : cells) {
    float rcalib = 1.;
    int granularity = 1;
    int threshold = 0;

    if (cell.isHcalTrigTowerDetId()) {
      rcalib = 0.;
    }

    HcalLutMetadatum item(cell.rawId(), rcalib, granularity, threshold);
    result->addValues(item);
  }

  return result;
}

std::unique_ptr<HcalDcsValues> HcalHardcodeCalibrations::produceDcsValues(const HcalDcsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceDcsValues-> ...";
  auto result = std::make_unique<HcalDcsValues>();
  return result;
}

std::unique_ptr<HcalDcsMap> HcalHardcodeCalibrations::produceDcsMap(const HcalDcsMapRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceDcsMap-> ...";

  return dbHardcode.makeHardcodeDcsMap();
}

std::unique_ptr<HcalRecoParams> HcalHardcodeCalibrations::produceRecoParams(const HcalRecoParamsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceRecoParams-> ...";
  auto const& topo = rec.get(topoTokens_[kRecoParams]);

  auto result = std::make_unique<HcalRecoParams>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalRecoParam item = dbHardcode.makeRecoParam(cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalTimingParams> HcalHardcodeCalibrations::produceTimingParams(const HcalTimingParamsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTimingParams-> ...";
  auto const& topo = rec.get(topoTokens_[kTimingParams]);

  auto result = std::make_unique<HcalTimingParams>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalTimingParam item = dbHardcode.makeTimingParam(cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalLongRecoParams> HcalHardcodeCalibrations::produceLongRecoParams(const HcalLongRecoParamsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceLongRecoParams-> ...";
  auto const& topo = rec.get(topoTokens_[kLongRecoParams]);

  auto result = std::make_unique<HcalLongRecoParams>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  std::vector<unsigned int> mSignal;
  mSignal.push_back(4);
  mSignal.push_back(5);
  mSignal.push_back(6);
  std::vector<unsigned int> mNoise;
  mNoise.push_back(1);
  mNoise.push_back(2);
  mNoise.push_back(3);
  for (auto cell : cells) {
    if (cell.isHcalZDCDetId()) {
      HcalLongRecoParam item(cell.rawId(), mSignal, mNoise);
      result->addValues(item);
    }
  }
  return result;
}

std::unique_ptr<HcalZDCLowGainFractions> HcalHardcodeCalibrations::produceZDCLowGainFractions(
    const HcalZDCLowGainFractionsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceZDCLowGainFractions-> ...";
  auto const& topo = rec.get(topoTokens_[kZDCLowGainFractions]);

  auto result = std::make_unique<HcalZDCLowGainFractions>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalZDCLowGainFraction item(cell.rawId(), 0.0);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalMCParams> HcalHardcodeCalibrations::produceMCParams(const HcalMCParamsRcd& rec) {
  //  std::cout << std::endl << " .... HcalHardcodeCalibrations::produceMCParams ->"<< std::endl;

  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceMCParams-> ...";
  auto const& topo = rec.get(topoTokens_[kMCParams]);
  auto result = std::make_unique<HcalMCParams>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalMCParam item = dbHardcode.makeMCParam(cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalFlagHFDigiTimeParams> HcalHardcodeCalibrations::produceFlagHFDigiTimeParams(
    const HcalFlagHFDigiTimeParamsRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceFlagHFDigiTimeParams-> ...";
  auto const& topo = rec.get(topoTokens_[kFlagHFDigiTimeParams]);

  auto result = std::make_unique<HcalFlagHFDigiTimeParams>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());

  std::vector<double> coef;
  coef.push_back(0.93);
  coef.push_back(-0.38275);
  coef.push_back(-0.012667);

  for (auto cell : cells) {
    HcalFlagHFDigiTimeParam item(cell.rawId(),
                                 1,    //firstsample
                                 3,    // samplestoadd
                                 2,    //expectedpeak
                                 40.,  // min energy threshold
                                 coef  // coefficients
    );
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalFrontEndMap> HcalHardcodeCalibrations::produceFrontEndMap(const HcalFrontEndMapRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceFrontEndMap-> ...";
  auto const& topo = rec.get(topoTokens_[kFrontEndMap]);

  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());

  return dbHardcode.makeHardcodeFrontEndMap(cells);
}

std::unique_ptr<HcalSiPMParameters> HcalHardcodeCalibrations::produceSiPMParameters(const HcalSiPMParametersRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceSiPMParameters-> ...";
  auto const& topo = rec.get(topoTokens_[kSiPMParameters]);

  auto result = std::make_unique<HcalSiPMParameters>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    HcalSiPMParameter item = dbHardcode.makeHardcodeSiPMParameter(cell, &topo, iLumi);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalSiPMCharacteristics> HcalHardcodeCalibrations::produceSiPMCharacteristics(
    const HcalSiPMCharacteristicsRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceSiPMCharacteristics-> ...";

  return dbHardcode.makeHardcodeSiPMCharacteristics();
}

std::unique_ptr<HcalTPChannelParameters> HcalHardcodeCalibrations::produceTPChannelParameters(
    const HcalTPChannelParametersRcd& rec) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTPChannelParameters-> ...";
  auto const& topo = rec.get(topoTokens_[kTPChannelParameters]);

  auto result = std::make_unique<HcalTPChannelParameters>(&topo);
  const std::vector<HcalGenericDetId>& cells = allCells(topo, dbHardcode.killHE());
  for (auto cell : cells) {
    // Thinking about Phase2 and the new FIR filter,
    // for now, don't put TT in TPChannelParams
    if (cell.subdetId() == HcalTriggerTower)
      continue;
    HcalTPChannelParameter item = dbHardcode.makeHardcodeTPChannelParameter(cell);
    result->addValues(item);
  }
  return result;
}

std::unique_ptr<HcalTPParameters> HcalHardcodeCalibrations::produceTPParameters(const HcalTPParametersRcd& rcd) {
  edm::LogInfo("HCAL") << "HcalHardcodeCalibrations::produceTPParameters-> ...";

  auto result = std::make_unique<HcalTPParameters>();
  dbHardcode.makeHardcodeTPParameters(*result);
  return result;
}

void HcalHardcodeCalibrations::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("iLumi", -1.);
  desc.add<bool>("HBRecalibration", false);
  desc.add<double>("HBreCalibCutoff", 20.);
  desc.add<edm::FileInPath>("HBmeanenergies", edm::FileInPath("CalibCalorimetry/HcalPlugins/data/meanenergiesHB.txt"));
  desc.add<bool>("HERecalibration", false);
  desc.add<double>("HEreCalibCutoff", 20.);
  desc.add<edm::FileInPath>("HEmeanenergies", edm::FileInPath("CalibCalorimetry/HcalPlugins/data/meanenergiesHE.txt"));
  desc.add<bool>("HFRecalibration", false);
  desc.add<bool>("GainWidthsForTrigPrims", false);
  desc.add<bool>("useHBUpgrade", false);
  desc.add<bool>("useHEUpgrade", false);
  desc.add<bool>("useHFUpgrade", false);
  desc.add<bool>("useHOUpgrade", true);
  desc.add<bool>("testHFQIE10", false);
  desc.add<bool>("testHEPlan1", false);
  desc.add<bool>("killHE", false);
  desc.add<bool>("useLayer0Weight", false);
  desc.add<bool>("useIeta18depth1", true);
  desc.addUntracked<std::vector<std::string>>("toGet", std::vector<std::string>());
  desc.addUntracked<bool>("fromDDD", false);

  edm::ParameterSetDescription desc_hb;
  desc_hb.add<std::vector<double>>("gain", std::vector<double>({0.19}));
  desc_hb.add<std::vector<double>>("gainWidth", std::vector<double>({0.0}));
  desc_hb.add<double>("pedestal", 3.0);
  desc_hb.add<double>("pedestalWidth", 0.55);
  desc_hb.add<int>("zsThreshold", 8);
  desc_hb.add<std::vector<double>>("qieOffset", std::vector<double>({-0.49, 1.8, 7.2, 37.9}));
  desc_hb.add<std::vector<double>>("qieSlope", std::vector<double>({0.912, 0.917, 0.922, 0.923}));
  desc_hb.add<int>("qieType", 0);
  desc_hb.add<int>("mcShape", 125);
  desc_hb.add<int>("recoShape", 105);
  desc_hb.add<double>("photoelectronsToAnalog", 0.0);
  desc_hb.add<std::vector<double>>("darkCurrent", std::vector<double>({0.0}));
  desc_hb.add<std::vector<double>>("noiseCorrelation", std::vector<double>({0.0}));
  desc_hb.add<bool>("doRadiationDamage", false);
  desc_hb.add<double>("noiseThreshold", 0.0);
  desc_hb.add<double>("seedThreshold", 0.1);
  desc.add<edm::ParameterSetDescription>("hb", desc_hb);

  edm::ParameterSetDescription desc_hbRaddam;
  desc_hbRaddam.add<double>("temperatureBase", 20.0);
  desc_hbRaddam.add<double>("temperatureNew", -5.0);
  desc_hbRaddam.add<double>("intlumiOffset", 150);
  desc_hbRaddam.add<double>("depVsTemp", 0.0631);
  desc_hbRaddam.add<double>("intlumiToNeutrons", 3.67e8);
  desc_hbRaddam.add<std::vector<double>>("depVsNeutrons", {5.69e-11, 7.90e-11});

  edm::ParameterSetDescription desc_hbUpgrade;
  desc_hbUpgrade.add<std::vector<double>>("gain", std::vector<double>({0.00111111111111}));
  desc_hbUpgrade.add<std::vector<double>>("gainWidth", std::vector<double>({0}));
  desc_hbUpgrade.add<double>("pedestal", 18.0);
  desc_hbUpgrade.add<double>("pedestalWidth", 5.0);
  desc_hbUpgrade.add<int>("zsThreshold", 3);
  desc_hbUpgrade.add<std::vector<double>>("qieOffset", std::vector<double>({0.0, 0.0, 0.0, 0.0}));
  desc_hbUpgrade.add<std::vector<double>>("qieSlope", std::vector<double>({0.333, 0.333, 0.333, 0.333}));
  desc_hbUpgrade.add<int>("qieType", 2);
  desc_hbUpgrade.add<int>("mcShape", 206);
  desc_hbUpgrade.add<int>("recoShape", 206);
  desc_hbUpgrade.add<double>("photoelectronsToAnalog", 57.5);
  desc_hbUpgrade.add<std::vector<double>>("darkCurrent", std::vector<double>({0.055}));
  desc_hbUpgrade.add<std::vector<double>>("noiseCorrelation", std::vector<double>({0.26}));
  desc_hbUpgrade.add<bool>("doRadiationDamage", true);
  desc_hbUpgrade.add<edm::ParameterSetDescription>("radiationDamage", desc_hbRaddam);
  desc_hbUpgrade.add<double>("noiseThreshold", 0.0);
  desc_hbUpgrade.add<double>("seedThreshold", 0.1);
  desc.add<edm::ParameterSetDescription>("hbUpgrade", desc_hbUpgrade);

  edm::ParameterSetDescription desc_he;
  desc_he.add<std::vector<double>>("gain", std::vector<double>({0.23}));
  desc_he.add<std::vector<double>>("gainWidth", std::vector<double>({0}));
  desc_he.add<double>("pedestal", 3.0);
  desc_he.add<double>("pedestalWidth", 0.79);
  desc_he.add<int>("zsThreshold", 9);
  desc_he.add<std::vector<double>>("qieOffset", std::vector<double>({-0.38, 2.0, 7.6, 39.6}));
  desc_he.add<std::vector<double>>("qieSlope", std::vector<double>({0.912, 0.916, 0.92, 0.922}));
  desc_he.add<int>("qieType", 0);
  desc_he.add<int>("mcShape", 125);
  desc_he.add<int>("recoShape", 105);
  desc_he.add<double>("photoelectronsToAnalog", 0.0);
  desc_he.add<std::vector<double>>("darkCurrent", std::vector<double>({0.0}));
  desc_he.add<std::vector<double>>("noiseCorrelation", std::vector<double>({0.0}));
  desc_he.add<bool>("doRadiationDamage", false);
  desc_he.add<double>("noiseThreshold", 0.0);
  desc_he.add<double>("seedThreshold", 0.1);
  desc.add<edm::ParameterSetDescription>("he", desc_he);

  edm::ParameterSetDescription desc_heRaddam;
  desc_heRaddam.add<double>("temperatureBase", 20.0);
  desc_heRaddam.add<double>("temperatureNew", 5.0);
  desc_heRaddam.add<double>("intlumiOffset", 75);
  desc_heRaddam.add<double>("depVsTemp", 0.0631);
  desc_heRaddam.add<double>("intlumiToNeutrons", 2.92e8);
  desc_heRaddam.add<std::vector<double>>("depVsNeutrons", {5.69e-11, 7.90e-11});

  edm::ParameterSetDescription desc_heUpgrade;
  desc_heUpgrade.add<std::vector<double>>("gain", std::vector<double>({0.00111111111111}));
  desc_heUpgrade.add<std::vector<double>>("gainWidth", std::vector<double>({0}));
  desc_heUpgrade.add<double>("pedestal", 18.0);
  desc_heUpgrade.add<double>("pedestalWidth", 5.0);
  desc_heUpgrade.add<int>("zsThreshold", 3);
  desc_heUpgrade.add<std::vector<double>>("qieOffset", std::vector<double>({0.0, 0.0, 0.0, 0.0}));
  desc_heUpgrade.add<std::vector<double>>("qieSlope", std::vector<double>({0.333, 0.333, 0.333, 0.333}));
  desc_heUpgrade.add<int>("qieType", 2);
  desc_heUpgrade.add<int>("mcShape", 206);
  desc_heUpgrade.add<int>("recoShape", 206);
  desc_heUpgrade.add<double>("photoelectronsToAnalog", 57.5);
  desc_heUpgrade.add<std::vector<double>>("darkCurrent", std::vector<double>({0.055}));
  desc_heUpgrade.add<std::vector<double>>("noiseCorrelation", std::vector<double>({0.26}));
  desc_heUpgrade.add<bool>("doRadiationDamage", true);
  desc_heUpgrade.add<edm::ParameterSetDescription>("radiationDamage", desc_heRaddam);
  desc_heUpgrade.add<double>("noiseThreshold", 0.0);
  desc_heUpgrade.add<double>("seedThreshold", 0.1);
  desc.add<edm::ParameterSetDescription>("heUpgrade", desc_heUpgrade);

  edm::ParameterSetDescription desc_hf;
  desc_hf.add<std::vector<double>>("gain", std::vector<double>({0.14, 0.135}));
  desc_hf.add<std::vector<double>>("gainWidth", std::vector<double>({0.0, 0.0}));
  desc_hf.add<double>("pedestal", 3.0);
  desc_hf.add<double>("pedestalWidth", 0.84);
  desc_hf.add<int>("zsThreshold", -9999);
  desc_hf.add<std::vector<double>>("qieOffset", std::vector<double>({-0.87, 1.4, 7.8, -29.6}));
  desc_hf.add<std::vector<double>>("qieSlope", std::vector<double>({0.359, 0.358, 0.36, 0.367}));
  desc_hf.add<int>("qieType", 0);
  desc_hf.add<int>("mcShape", 301);
  desc_hf.add<int>("recoShape", 301);
  desc_hf.add<double>("photoelectronsToAnalog", 0.0);
  desc_hf.add<std::vector<double>>("darkCurrent", std::vector<double>({0.0}));
  desc_hf.add<std::vector<double>>("noiseCorrelation", std::vector<double>({0.0}));
  desc_hf.add<bool>("doRadiationDamage", false);
  desc_hf.add<double>("noiseThreshold", 0.0);
  desc_hf.add<double>("seedThreshold", 0.1);
  desc.add<edm::ParameterSetDescription>("hf", desc_hf);

  edm::ParameterSetDescription desc_hfUpgrade;
  desc_hfUpgrade.add<std::vector<double>>("gain", std::vector<double>({0.14, 0.135}));
  desc_hfUpgrade.add<std::vector<double>>("gainWidth", std::vector<double>({0.0, 0.0}));
  desc_hfUpgrade.add<double>("pedestal", 13.33);
  desc_hfUpgrade.add<double>("pedestalWidth", 3.33);
  desc_hfUpgrade.add<int>("zsThreshold", -9999);
  desc_hfUpgrade.add<std::vector<double>>("qieOffset", std::vector<double>({0.0697, -0.7405, 12.38, -671.9}));
  desc_hfUpgrade.add<std::vector<double>>("qieSlope", std::vector<double>({0.297, 0.298, 0.298, 0.313}));
  desc_hfUpgrade.add<int>("qieType", 1);
  desc_hfUpgrade.add<int>("mcShape", 301);
  desc_hfUpgrade.add<int>("recoShape", 301);
  desc_hfUpgrade.add<double>("photoelectronsToAnalog", 0.0);
  desc_hfUpgrade.add<std::vector<double>>("darkCurrent", std::vector<double>({0.0}));
  desc_hfUpgrade.add<std::vector<double>>("noiseCorrelation", std::vector<double>({0.0}));
  desc_hfUpgrade.add<bool>("doRadiationDamage", false);
  desc_hfUpgrade.add<double>("noiseThreshold", 0.0);
  desc_hfUpgrade.add<double>("seedThreshold", 0.1);
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
  desc_ho.add<int>("zsThreshold", 24);
  desc_ho.add<std::vector<double>>("qieOffset", std::vector<double>({-0.44, 1.4, 7.1, 38.5}));
  desc_ho.add<std::vector<double>>("qieSlope", std::vector<double>({0.907, 0.915, 0.92, 0.921}));
  desc_ho.add<int>("qieType", 0);
  desc_ho.add<int>("mcShape", 201);
  desc_ho.add<int>("recoShape", 201);
  desc_ho.add<double>("photoelectronsToAnalog", 4.0);
  desc_ho.add<std::vector<double>>("darkCurrent", std::vector<double>({0.0}));
  desc_ho.add<std::vector<double>>("noiseCorrelation", std::vector<double>({0.0}));
  desc_ho.add<bool>("doRadiationDamage", false);
  desc_ho.add<double>("noiseThreshold", 0.0);
  desc_ho.add<double>("seedThreshold", 0.1);
  desc.add<edm::ParameterSetDescription>("ho", desc_ho);

  edm::ParameterSetDescription validator_sipm;
  validator_sipm.add<int>("pixels", 1);
  validator_sipm.add<double>("crosstalk", 0);
  validator_sipm.add<double>("nonlin1", 1);
  validator_sipm.add<double>("nonlin2", 0);
  validator_sipm.add<double>("nonlin3", 0);
  std::vector<edm::ParameterSet> default_sipm(1);
  desc.addVPSet("SiPMCharacteristics", validator_sipm, default_sipm);

  descriptions.addDefault(desc);
}
