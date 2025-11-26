/*
 * InputMakerPhase2.cpp
 *
 *  Created on: May 20, 2020
 *      Author: kbunkow
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/InputMakerPhase2.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OmtfName.h"

#include <iostream>

/////////////////////////////////////
void DtPhase2DigiToStubsConverter::loadDigis(const edm::Event& event) {
  event.getByToken(inputTokenDtPh, dtPhDigis);
  event.getByToken(inputTokenDtTh, dtThDigis);
  bunchCrossing = event.bunchCrossing();
}

void DtPhase2DigiToStubsConverter::makeStubs(MuonStubPtrs2D& muonStubsInLayers,
                                             unsigned int iProcessor,
                                             l1t::tftype procTyp,
                                             int bxFrom,
                                             int bxTo,
                                             std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  boost::property_tree::ptree procDataTree;

  std::map<std::string, boost::property_tree::ptree> chamberTrees;

  for (const auto& digiIt : *dtPhDigis->getContainer()) {
    DTChamberId detid(digiIt.whNum(), digiIt.stNum(), digiIt.scNum() + 1);

    LogTrace("l1tOmtfEventPrint") << " L1Phase2MuDTPhDigi:35: detid " << detid << " digi "
                                  << " whNum " << digiIt.whNum() << " scNum " << digiIt.scNum() << " stNum "
                                  << digiIt.stNum() << " slNum " << digiIt.slNum() << " quality " << digiIt.quality()
                                  << " rpcFlag " << digiIt.rpcFlag() << " phi " << digiIt.phi() << " phiBend "
                                  << digiIt.phiBend() << " digBx " << digiIt.bxNum() << " correctedBx "
                                  << digiIt.bxNum() - bunchCrossing << std::endl;

    ///Check it the data fits into given processor input range
    if (!acceptDigi(detid, iProcessor, procTyp))
      continue;

    // HACK for Phase-2  (DT TPs are centered in bX=20 in MC, and 13(?) in data)
    if (digiIt.bxNum() - config->dtBxShift() >= bxFrom && digiIt.bxNum() - config->dtBxShift() <= bxTo) {
      addDTphiDigi(muonStubsInLayers, digiIt, dtThDigis.product(), iProcessor, procTyp);

      std::ostringstream chamberName;
      //chamberName<<detid;
      chamberName << "DT_Wh" << digiIt.whNum() << "_St_" << digiIt.stNum() << "_Se_" << digiIt.scNum() + 1;
      auto& dtChamberTree = chamberTrees[chamberName.str()];
      auto& dtP2PhiDigi = dtChamberTree.add_child("dtP2PhiDigi", boost::property_tree::ptree());
      dtP2PhiDigi.add("<xmlattr>.whNum", digiIt.whNum());
      dtP2PhiDigi.add("<xmlattr>.scNum", digiIt.scNum());
      dtP2PhiDigi.add("<xmlattr>.stNum", digiIt.stNum());
      dtP2PhiDigi.add("<xmlattr>.slNum", digiIt.slNum());
      dtP2PhiDigi.add("<xmlattr>.quality", digiIt.quality());
      dtP2PhiDigi.add("<xmlattr>.rpcFlag", digiIt.rpcFlag());
      dtP2PhiDigi.add("<xmlattr>.phi", digiIt.phi());
      dtP2PhiDigi.add("<xmlattr>.phiBend", digiIt.phiBend());
    }
  }

  for (auto& thetaDigi : (*(dtThDigis->getContainer()))) {
    if (thetaDigi.bxNum() - config->dtBxShift() >= bxFrom && thetaDigi.bxNum() - config->dtBxShift() <= bxTo) {
      if (!mergePhiAndTheta) {
        addDTetaStubs(muonStubsInLayers, thetaDigi, iProcessor, procTyp);
      }

      std::ostringstream chamberName;
      //chamberName<<detid;
      chamberName << "DT_Wh" << thetaDigi.whNum() << "_St_" << thetaDigi.stNum() << "_Se_" << thetaDigi.scNum() + 1;
      auto& dtChamberTree = chamberTrees[chamberName.str()];
      auto& dtP2ThDigi = dtChamberTree.add_child("dtP2ThDigi", boost::property_tree::ptree());
      dtP2ThDigi.add("<xmlattr>.whNum", thetaDigi.whNum());
      dtP2ThDigi.add("<xmlattr>.scNum", thetaDigi.scNum());
      dtP2ThDigi.add("<xmlattr>.stNum", thetaDigi.stNum());
      dtP2ThDigi.add("<xmlattr>.quality", thetaDigi.quality());
      dtP2ThDigi.add("<xmlattr>.rpcFlag", thetaDigi.rpcFlag());
      dtP2ThDigi.add("<xmlattr>.k", thetaDigi.k());
      dtP2ThDigi.add("<xmlattr>.z", thetaDigi.z());
    }
  }

  for (auto& chamberTree : chamberTrees) {
    chamberTree.second.add("<xmlattr>.name", chamberTree.first);
    procDataTree.add_child("dtChamber", chamberTree.second);
  }

  for (auto& obs : observers)
    obs->addProcesorData("dtData", procDataTree);
}

//dtThDigis is provided as argument, because in the OMTF implementation the phi and eta digis are merged (even thought it is artificial)
void DtPhase2DigiToStubsConverterOmtf::addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers,
                                                    const L1Phase2MuDTPhDigi& digi,
                                                    const L1Phase2MuDTThContainer* dtThDigis,
                                                    unsigned int iProcessor,
                                                    l1t::tftype procTyp) {
  DTChamberId detid(digi.whNum(), digi.stNum(), digi.scNum() + 1);

  MuonStub stub;

  //converting the quality to the same encoding as in phase-1, as it is important for extrapolation
  if (digi.quality() >= 6)  // correlated stub
    stub.qualityHw = digi.quality() - 2;
  else if (digi.quality() >= 3) {  // 4 hit uncorrelated stub
    if (digi.slNum() == 3)
      stub.qualityHw = 3;
    else if (digi.slNum() == 1)
      stub.qualityHw = 2;
  } else {  //quality 1 (3 hits) or 2 (3+2 hits)
    if (digi.slNum() == 3)
      stub.qualityHw = 1;
    else if (digi.slNum() == 1)
      stub.qualityHw = 0;
  }

  if (stub.qualityHw < config->getMinDtPhiQuality())
    return;

  unsigned int hwNumber = config->getLayerNumber(detid.rawId());
  if (config->getHwToLogicLayer().find(hwNumber) == config->getHwToLogicLayer().end())
    return;

  auto iter = config->getHwToLogicLayer().find(hwNumber);
  unsigned int iLayer = iter->second;
  unsigned int iInput = OMTFinputMaker::getInputNumber(config, detid.rawId(), iProcessor, procTyp);
  //MuonStub& stub = muonStubsInLayers[iLayer][iInput];

  stub.type = MuonStub::DT_PHI_ETA;

  stub.phiHw = angleConverter.getProcessorPhi(
      OMTFinputMaker::getProcessorPhiZero(config, iProcessor), procTyp, digi.scNum(), digi.phi());

  //no config->dtBxShift() here, and also no shift inside getGlobalEtaPhase2
  stub.etaHw = angleConverter.getGlobalEtaPhase2(detid, dtThDigis, digi.bxNum());

  if (iLayer == 0)
    stub.r = 431.175;  //MB1
  else if (iLayer == 2) {
    stub.r = 512.475;  //MB2
  } else if (iLayer == 4) {
    stub.r = 620;  //round(619.675);
    //MB3, it is different than in the phase-1, as in the phase-2 it is a middle of the DT chamber, not muon station
  }

  //in phase2, the phiB is 13 bits, and range is [-2, 2 rad] so 4 rad, 2^13 units/(4 rad) =  1^11/rad.
  //need to convert them to 512units==1rad (to use OLD PATTERNS...)
  stub.phiBHw = digi.phiBend() * config->dtPhiBUnitsRad() / 2048;
  //the cut if (stub.qualityHw >= config.getMinDtPhiBQuality()) is done in the ProcessorBase<GoldenPatternType>::restrictInput
  //as is is done like that in the firmware

  // need to shift 20-BX to roll-back the shift introduced by the DT TPs
  stub.bx = digi.bxNum() - config->dtBxShift();
  //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?

  stub.logicLayer = iLayer;
  stub.detId = detid;

  OmtfName board(iProcessor, config);
  LogTrace("l1tOmtfEventPrint") << board.name() << " L1Phase2MuDTPhDigi: detid " << detid << " digi "
                                << " whNum " << digi.whNum() << " scNum " << digi.scNum() << " stNum " << digi.stNum()
                                << " slNum " << digi.slNum() << " quality " << digi.quality() << " rpcFlag "
                                << digi.rpcFlag() << " phi " << digi.phi() << " phiBend " << digi.phiBend() << " bx "
                                << digi.bxNum() << std::endl;
  LogTrace("l1tOmtfEventPrint") << board.name() << " stub: detid " << detid << " phi " << stub.phiHw << " eta "
                                << stub.etaHw << " phiB " << stub.phiBHw << " bx " << stub.bx << " quality "
                                << stub.qualityHw << " logicLayer " << stub.logicLayer << std::endl;
  OMTFinputMaker::addStub(config, muonStubsInLayers, iLayer, iInput, stub);
}

void DtPhase2DigiToStubsConverterOmtf::addDTetaStubs(MuonStubPtrs2D& muonStubsInLayers,
                                                     const L1Phase2MuDTThDigi& thetaDigi,
                                                     unsigned int iProcessor,
                                                     l1t::tftype procTyp) {
  //in the Phase1 omtf the theta stubs are merged with the phi in the addDTphiDigi
  //TODO implement if needed
}

bool DtPhase2DigiToStubsConverterOmtf::acceptDigi(const DTChamberId& dTChamberId,
                                                  unsigned int iProcessor,
                                                  l1t::tftype procType) {
  return OMTFinputMaker::acceptDtDigi(config, dTChamberId, iProcessor, procType);
}

InputMakerPhase2::InputMakerPhase2(const edm::ParameterSet& edmParameterSet,
                                   MuStubsInputTokens& muStubsInputTokens,
                                   MuStubsPhase2InputTokens& muStubsPhase2InputTokens,
                                   const OMTFConfiguration* config,
                                   std::unique_ptr<OmtfAngleConverter> angleConverter)
    : OMTFinputMaker(edmParameterSet, muStubsInputTokens, config, std::move(angleConverter)) {
  edm::LogImportant("OMTFReconstruction") << "constructing InputMakerPhase2" << std::endl;

  if (edmParameterSet.exists("usePhase2DTPrimitives") && edmParameterSet.getParameter<bool>("usePhase2DTPrimitives")) {
    if (edmParameterSet.getParameter<bool>("dropDTPrimitives") != true)
      throw cms::Exception(
          "L1TMuonOverlapPhase2 InputMakerPhase2::InputMakerPhase2 usePhase2DTPrimitives is true, but dropDTPrimitives "
          "is not true");
    //if the Phase2DTPrimitives are used, then the phase1 DT primitives should be dropped
    edm::LogImportant("OMTFReconstruction") << " using Phase2 DT trigger primitives" << std::endl;

    digiToStubsConverters.emplace_back(std::make_unique<DtPhase2DigiToStubsConverterOmtf>(
        config,
        dynamic_cast<OmtfPhase2AngleConverter*>(this->angleConverter.get()),
        muStubsPhase2InputTokens.inputTokenDtPh,
        muStubsPhase2InputTokens.inputTokenDtTh));
  }
}
