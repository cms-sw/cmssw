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
}

void DtPhase2DigiToStubsConverter::makeStubs(MuonStubPtrs2D& muonStubsInLayers,
                                             unsigned int iProcessor,
                                             l1t::tftype procTyp,
                                             int bxFrom,
                                             int bxTo,
                                             std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  if (!dtPhDigis)
    return;

  boost::property_tree::ptree procDataTree;

  for (const auto& digiIt : *dtPhDigis->getContainer()) {
    DTChamberId detid(digiIt.whNum(), digiIt.stNum(), digiIt.scNum() + 1);

    ///Check it the data fits into given processor input range
    if (!acceptDigi(detid, iProcessor, procTyp))
      continue;

    // HACK for Phase-2  (DT TPs are centered in bX=20)
    if (digiIt.bxNum() - 20 >= bxFrom && digiIt.bxNum() - 20 <= bxTo) {
      addDTphiDigi(muonStubsInLayers, digiIt, dtThDigis.product(), iProcessor, procTyp);

      auto& dtP2Digi = procDataTree.add_child("dtP2Digi", boost::property_tree::ptree());
      dtP2Digi.add("<xmlattr>.whNum", digiIt.whNum());
      dtP2Digi.add("<xmlattr>.scNum", digiIt.scNum());
      dtP2Digi.add("<xmlattr>.stNum", digiIt.stNum());
      dtP2Digi.add("<xmlattr>.slNum", digiIt.slNum());
      dtP2Digi.add("<xmlattr>.quality", digiIt.quality());
      dtP2Digi.add("<xmlattr>.rpcFlag", digiIt.rpcFlag());
      dtP2Digi.add("<xmlattr>.phi", digiIt.phi());
      dtP2Digi.add("<xmlattr>.phiBend", digiIt.phiBend());
    }
  }

  if (!mergePhiAndTheta) {
    for (auto& thetaDigi : (*(dtThDigis->getContainer()))) {
      if (thetaDigi.bxNum() >= bxFrom && thetaDigi.bxNum() <= bxTo) {
        addDTetaStubs(muonStubsInLayers, thetaDigi, iProcessor, procTyp);
      }
    }
  }

  for (auto& obs : observers)
    obs->addProcesorData("linkData", procDataTree);
}

//dtThDigis is provided as argument, because in the OMTF implementation the phi and eta digis are merged (even thought it is artificial)
void DtPhase2DigiToStubsConverterOmtf::addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers,
                                                    const L1Phase2MuDTPhDigi& digi,
                                                    const L1MuDTChambThContainer* dtThDigis,
                                                    unsigned int iProcessor,
                                                    l1t::tftype procTyp) {
  DTChamberId detid(digi.whNum(), digi.stNum(), digi.scNum() + 1);

  MuonStub stub;

  //converting the quality to the same encoding as in phase-1, as it is important for extrapolation
  if (digi.quality() >= 6)
    stub.qualityHw = digi.quality() - 2;
  else if (digi.quality() >= 3) {
    if (digi.slNum() == 3)
      stub.qualityHw = 3;
    else if (digi.slNum() == 1)
      stub.qualityHw = 2;
  } else {
    if (digi.slNum() == 3)
      stub.qualityHw = 1;
    else if (digi.slNum() == 1)
      stub.qualityHw = 0;
  }

  if (stub.qualityHw < config.getMinDtPhiQuality())
    return;

  unsigned int hwNumber = config.getLayerNumber(detid.rawId());
  if (config.getHwToLogicLayer().find(hwNumber) == config.getHwToLogicLayer().end())
    return;

  auto iter = config.getHwToLogicLayer().find(hwNumber);
  unsigned int iLayer = iter->second;
  unsigned int iInput = OMTFinputMaker::getInputNumber(&config, detid.rawId(), iProcessor, procTyp);
  //MuonStub& stub = muonStubsInLayers[iLayer][iInput];

  stub.type = MuonStub::DT_PHI_ETA;

  stub.phiHw = angleConverter.getProcessorPhi(
      OMTFinputMaker::getProcessorPhiZero(&config, iProcessor), procTyp, digi.scNum(), digi.phi());

  //TODO the dtThDigis are not good yet,so passing an empty container to the angleConverter
  //then it should return middle of chambers
  //remove when the dtThDigis are fixed on the DT side
  L1MuDTChambThContainer dtThDigisEmpty;
  stub.etaHw = angleConverter.getGlobalEta(detid, &dtThDigisEmpty, digi.bxNum() - 20);
  //in phase2, the phiB is 13 bits, and range is [-2, 2 rad] so 4 rad, 2^13 units/(4 rad) =  1^11/rad.
  //need to convert them to 512units==1rad (to use OLD PATTERNS...)
  stub.phiBHw = digi.phiBend() * config.dtPhiBUnitsRad() / 2048;
  //the cut if (stub.qualityHw >= config.getMinDtPhiBQuality()) is done in the ProcessorBase<GoldenPatternType>::restrictInput
  //as is is done like that in the firmware

  // need to shift 20-BX to roll-back the shift introduced by the DT TPs
  stub.bx = digi.bxNum() - 20;
  //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?

  stub.logicLayer = iLayer;
  stub.detId = detid;

  OmtfName board(iProcessor, &config);
  LogTrace("l1tOmtfEventPrint") << board.name() << " L1Phase2MuDTPhDigi: detid " << detid << " digi "
                                << " whNum " << digi.whNum() << " scNum " << digi.scNum() << " stNum " << digi.stNum()
                                << " slNum " << digi.slNum() << " quality " << digi.quality() << " rpcFlag "
                                << digi.rpcFlag() << " phi " << digi.phi() << " phiBend " << digi.phiBend()
                                << std::endl;
  OMTFinputMaker::addStub(&config, muonStubsInLayers, iLayer, iInput, stub);
}

void DtPhase2DigiToStubsConverterOmtf::addDTetaStubs(MuonStubPtrs2D& muonStubsInLayers,
                                                     const L1MuDTChambThDigi& thetaDigi,
                                                     unsigned int iProcessor,
                                                     l1t::tftype procTyp) {
  //in the Phase1 omtf the theta stubs are merged with the phi in the addDTphiDigi
  //TODO implement if needed
}

bool DtPhase2DigiToStubsConverterOmtf::acceptDigi(const DTChamberId& dTChamberId,
                                                  unsigned int iProcessor,
                                                  l1t::tftype procType) {
  return OMTFinputMaker::acceptDtDigi(&config, dTChamberId, iProcessor, procType);
}

InputMakerPhase2::InputMakerPhase2(const edm::ParameterSet& edmParameterSet,
                                   MuStubsInputTokens& muStubsInputTokens,
                                   edm::EDGetTokenT<L1Phase2MuDTPhContainer> inputTokenDTPhPhase2,
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
        config, this->angleConverter.get(), inputTokenDTPhPhase2, muStubsInputTokens.inputTokenDtTh));
  }
}
