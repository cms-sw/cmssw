#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStubMakerBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/ProcConfigurationBase.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iterator>
#include <utility>

/////////////////////////////////////
void DtDigiToStubsConverter::loadDigis(const edm::Event& event) {
  event.getByToken(inputTokenDtPh, dtPhDigis);
  event.getByToken(inputTokenDtTh, dtThDigis);
}

void DtDigiToStubsConverter::makeStubs(MuonStubPtrs2D& muonStubsInLayers,
                                       unsigned int iProcessor,
                                       l1t::tftype procTyp,
                                       int bxFrom,
                                       int bxTo,
                                       std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  for (const auto& digiIt : *dtPhDigis->getContainer()) {
    DTChamberId detid(digiIt.whNum(), digiIt.stNum(), digiIt.scNum() + 1);

    ///Check it the data fits into given processor input range
    if (!acceptDigi(detid, iProcessor, procTyp))
      continue;

    if (digiIt.bxNum() >= bxFrom && digiIt.bxNum() <= bxTo)
      addDTphiDigi(muonStubsInLayers, digiIt, dtThDigis.product(), iProcessor, procTyp);
  }

  if (!mergePhiAndTheta) {
    for (auto& thetaDigi : (*(dtThDigis->getContainer()))) {
      if (thetaDigi.bxNum() >= bxFrom && thetaDigi.bxNum() <= bxTo) {
        addDTetaStubs(muonStubsInLayers, thetaDigi, iProcessor, procTyp);
      }
    }
  }
  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" iProcessor "<<iProcessor<<std::endl;
}
///////////////////////////////////////
///////////////////////////////////////

void CscDigiToStubsConverter::makeStubs(MuonStubPtrs2D& muonStubsInLayers,
                                        unsigned int iProcessor,
                                        l1t::tftype procTyp,
                                        int bxFrom,
                                        int bxTo,
                                        std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  auto chamber = cscDigis->begin();
  auto chend = cscDigis->end();
  for (; chamber != chend; ++chamber) {
    unsigned int rawid = (*chamber).first;
    ///Check it the data fits into given processor input range
    CSCDetId csc(rawid);
    if (!acceptDigi(csc, iProcessor, procTyp))
      continue;

    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for (; digi != dend; ++digi) {
      ///Check if LCT trigger primitive has the right BX.
      int digiBx = digi->getBX() - config->cscLctCentralBx();

      if (digiBx >= bxFrom && digiBx <= bxTo)
        addCSCstubs(muonStubsInLayers, rawid, *digi, iProcessor, procTyp);
    }
  }
}

void RpcDigiToStubsConverter::makeStubs(MuonStubPtrs2D& muonStubsInLayers,
                                        unsigned int iProcessor,
                                        l1t::tftype procTyp,
                                        int bxFrom,
                                        int bxTo,
                                        std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  //LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ <<" RPC HITS, processor : " << iProcessor<<" "<<std::endl;

  const RPCDigiCollection& rpcDigiCollection = *rpcDigis;
  for (auto rollDigis : rpcDigiCollection) {
    RPCDetId roll = rollDigis.first;

    //debug
    //if(roll.region() != 0  &&  abs(roll.station()) >= 3 && roll.ring() == 1 )
    /*    {
      //iRPC
      for (auto pDigi=rollDigis.second.first; pDigi != rollDigis.second.second; pDigi++) {
        LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" roll "<<roll
            <<" strip "<<pDigi->strip()
            <<" hasX "<<pDigi->hasX()<<" coordinateX "<<pDigi->coordinateX()<<" hasY "<<pDigi->hasY()<<" coordinateY "<<pDigi->coordinateY()
            <<" bx "<<pDigi->bx()<<" time "<<pDigi->time()<<" irpc"<<std::endl;
      }
      //continue;
    }*/

    //LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ <<" roll "<<roll<<" "<<std::endl;

    if (!acceptDigi(roll, iProcessor, procTyp))
      continue;

    ///To find the clusters we have to copy the digis in chamber to sort them (not optimal).
    //  for (auto tdigi = rollDigis.second.first; tdigi != rollDigis.second.second; tdigi++) { std::cout << "RPC DIGIS: " << roll.rawId()<< " "<<roll<<" digi: " << tdigi->strip() <<" bx: " << tdigi->bx() << std::endl; }
    std::vector<RPCDigi> digisCopy;

    for (auto pDigi = rollDigis.second.first; pDigi != rollDigis.second.second; pDigi++) {
      if (pDigi->bx() >= bxFrom && pDigi->bx() <= bxTo) {
        digisCopy.push_back(*pDigi);
      }
    }

    std::vector<RpcCluster> clusters = rpcClusterization->getClusters(roll, digisCopy);

    for (auto& cluster : clusters) {
      addRPCstub(muonStubsInLayers, roll, cluster, iProcessor, procTyp);
    }
  }

  //removing the RPC stubs that were mark as dropped in the RpcDigiToStubsConverterOmtf::addRPCstub
  //10 is the first RPC layer
  for (unsigned int iLayer = 10; iLayer < muonStubsInLayers.size(); iLayer++) {
    for (unsigned int iInput = 0; iInput < muonStubsInLayers[iLayer].size(); iInput++) {
      if (muonStubsInLayers[iLayer][iInput] && muonStubsInLayers[iLayer][iInput]->type == MuonStub::RPC_DROPPED) {
        LogTrace("l1tOmtfEventPrint") << "RpcDigiToStubsConverter::makeStubs "
                                      << " iProcessor " << iProcessor << " procTyp " << procTyp
                                      << " dropping a stub iLayer " << iLayer << " iInput "
                                      << *(muonStubsInLayers[iLayer][iInput]) << std::endl;
        muonStubsInLayers[iLayer][iInput].reset();
      }
    }
  }
}

///////////////////////////////////////
///////////////////////////////////////
MuonStubMakerBase::MuonStubMakerBase(const ProcConfigurationBase* procConf) : config(procConf), rpcClusterization() {}

///////////////////////////////////////
///////////////////////////////////////
void MuonStubMakerBase::initialize(const edm::ParameterSet& edmCfg,
                                   const edm::EventSetup& es,
                                   const MuonGeometryTokens& muonGeometryTokens) {
  rpcClusterization.configure(
      config->getRpcMaxClusterSize(), config->getRpcMaxClusterCnt(), config->getRpcDropAllClustersIfMoreThanMax());
}
///////////////////////////////////////
///////////////////////////////////////
MuonStubMakerBase::~MuonStubMakerBase() {}
///////////////////////////////////////
///////////////////////////////////////

void MuonStubMakerBase::loadAndFilterDigis(const edm::Event& event) {
  for (auto& digiToStubsConverter : digiToStubsConverters)
    digiToStubsConverter->loadDigis(event);
}

void MuonStubMakerBase::buildInputForProcessor(MuonStubPtrs2D& muonStubsInLayers,
                                               unsigned int iProcessor,
                                               l1t::tftype procTyp,
                                               int bxFrom,
                                               int bxTo,
                                               std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  //LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " iProcessor " << iProcessor << " preocType "
  //                              << procTyp << std::endl;

  for (auto& digiToStubsConverter : digiToStubsConverters)
    digiToStubsConverter->makeStubs(muonStubsInLayers, iProcessor, procTyp, bxFrom, bxTo, observers);
}
