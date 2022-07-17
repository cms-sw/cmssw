#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/ProcConfigurationBase.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <map>
#include <sstream>
#include <string>

//dtThDigis is provided as argument, because in the OMTF implementation the phi and eta digis are merged (even thought it is artificial)
void DtDigiToStubsConverterOmtf::addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers,
                                              const L1MuDTChambPhDigi& digi,
                                              const L1MuDTChambThContainer* dtThDigis,
                                              unsigned int iProcessor,
                                              l1t::tftype procTyp) {
  DTChamberId detid(digi.whNum(), digi.stNum(), digi.scNum() + 1);

  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" OMTFinputMaker "<<" detid "<<detid<<endl;
  ///Check Trigger primitive quality
  ///Ts2Tag() == 0 - take only first track from DT Trigger Server
  ///BxCnt()  == 0 - ??
  ///code()>=3     - take only double layer hits, HH, HL and LL
  // FIXME (MK): at least Ts2Tag selection is not correct! Check it
  //    if (digiIt.bxNum()!= 0 || digiIt.BxCnt()!= 0 || digiIt.Ts2Tag()!= 0 || digiIt.code()<4) continue;

  //7 is empty digi, TODO update if the definition of the quality is changed
  if (digi.code() == 7 || digi.code() < config->getMinDtPhiQuality())
    return;

  unsigned int hwNumber = config->getLayerNumber(detid.rawId());
  if (config->getHwToLogicLayer().find(hwNumber) == config->getHwToLogicLayer().end())
    return;

  auto iter = config->getHwToLogicLayer().find(hwNumber);
  unsigned int iLayer = iter->second;
  unsigned int iInput = OMTFinputMaker::getInputNumber(config, detid.rawId(), iProcessor, procTyp);
  //MuonStub& stub = muonStubsInLayers[iLayer][iInput];
  MuonStub stub;

  stub.type = MuonStub::DT_PHI_ETA;
  stub.qualityHw = digi.code();
  stub.phiHw = angleConverter->getProcessorPhi(
      OMTFinputMaker::getProcessorPhiZero(config, iProcessor), procTyp, digi.scNum(), digi.phi());
  stub.etaHw = angleConverter->getGlobalEta(detid, dtThDigis, digi.bxNum());

  if (stub.qualityHw >= config->getMinDtPhiBQuality())
    stub.phiBHw = digi.phiB();
  else
    stub.phiBHw = config->nPhiBins();

  stub.bx = digi.bxNum();  //TODO sholdn't  it be BxCnt()?
  //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?

  //stub.etaType = ?? TODO
  stub.logicLayer = iLayer;
  stub.detId = detid;

  OMTFinputMaker::addStub(config, muonStubsInLayers, iLayer, iInput, stub);
}

void DtDigiToStubsConverterOmtf::addDTetaStubs(MuonStubPtrs2D& muonStubsInLayers,
                                               const L1MuDTChambThDigi& thetaDigi,
                                               unsigned int iProcessor,
                                               l1t::tftype procTyp) {
  //in the Phase1 omtf the theta stubs are merged with the phi in the addDTphiDigi
}

bool DtDigiToStubsConverterOmtf::acceptDigi(const DTChamberId& dTChamberId,
                                            unsigned int iProcessor,
                                            l1t::tftype procType) {
  return OMTFinputMaker::acceptDtDigi(config, dTChamberId, iProcessor, procType);
}

void CscDigiToStubsConverterOmtf::addCSCstubs(MuonStubPtrs2D& muonStubsInLayers,
                                              unsigned int rawid,
                                              const CSCCorrelatedLCTDigi& digi,
                                              unsigned int iProcessor,
                                              l1t::tftype procTyp) {
  unsigned int hwNumber = config->getLayerNumber(rawid);
  if (config->getHwToLogicLayer().find(hwNumber) == config->getHwToLogicLayer().end())
    return;

  unsigned int iLayer = config->getHwToLogicLayer().at(hwNumber);
  unsigned int iInput = OMTFinputMaker::getInputNumber(config, rawid, iProcessor, procTyp);

  MuonStub stub;
  stub.type = MuonStub::CSC_PHI_ETA;
  stub.phiHw = angleConverter->getProcessorPhi(
      OMTFinputMaker::getProcessorPhiZero(config, iProcessor), procTyp, CSCDetId(rawid), digi);
  stub.etaHw = angleConverter->getGlobalEta(rawid, digi);
  stub.phiBHw = digi.getPattern();  //TODO change to phiB when implemented
  stub.qualityHw = digi.getQuality();

  stub.bx = digi.getBX() - config->cscLctCentralBx();  //TODO sholdn't  it be getBX0()?
  //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?

  //stub.etaType = ?? TODO
  stub.logicLayer = iLayer;
  stub.detId = rawid;

  OMTFinputMaker::addStub(config, muonStubsInLayers, iLayer, iInput, stub);
  ///Accept CSC digis only up to eta=1.26.
  ///The nominal OMTF range is up to 1.24, but cutting at 1.24
  ///kill efficiency at the edge. 1.26 is one eta bin above nominal.
  //if(abs(iEta)>1.26/2.61*240) continue;
  //if (abs(iEta) > 115) continue;

  //LogTrace("l1tOmtfEventPrint")<<" ADDING CSC hit, proc: "<<iProcessor<<" iPhi : " << iPhi <<" iEta: "<< iEta << std::endl;
}

bool CscDigiToStubsConverterOmtf::acceptDigi(const CSCDetId& csc, unsigned int iProcessor, l1t::tftype procType) {
  if (procType == l1t::tftype::omtf_pos && (csc.endcap() == 2 || csc.ring() == 1 || csc.station() == 4))
    return false;
  if (procType == l1t::tftype::omtf_neg && (csc.endcap() == 1 || csc.ring() == 1 || csc.station() == 4))
    return false;

  if (procType == l1t::tftype::emtf_pos && (csc.endcap() == 2 || (csc.station() == 1 && csc.ring() == 3)))
    return false;
  if (procType == l1t::tftype::emtf_neg && (csc.endcap() == 1 || (csc.station() == 1 && csc.ring() == 3)))
    return false;

  unsigned int aSector = csc.chamber();
  unsigned int aMin = config->getEndcap10DegMin()[iProcessor];
  unsigned int aMax = config->getEndcap10DegMax()[iProcessor];

  if ((procType == l1t::tftype::emtf_pos || procType == l1t::tftype::emtf_neg) && csc.station() > 1 &&
      csc.ring() == 1) {
    aMin = config->getEndcap20DegMin()[iProcessor];
    aMax = config->getEndcap20DegMax()[iProcessor];
  }

  if (aMax > aMin && aSector >= aMin && aSector <= aMax)
    return true;
  if (aMax < aMin && (aSector >= aMin || aSector <= aMax))
    return true;

  return false;
}

void RpcDigiToStubsConverterOmtf::addRPCstub(MuonStubPtrs2D& muonStubsInLayers,
                                             const RPCDetId& roll,
                                             const RpcCluster& cluster,
                                             unsigned int iProcessor,
                                             l1t::tftype procTyp) {
  unsigned int rawid = roll.rawId();

  unsigned int hwNumber = config->getLayerNumber(rawid);
  unsigned int iLayer = config->getHwToLogicLayer().at(hwNumber);
  unsigned int iInput = OMTFinputMaker::getInputNumber(config, rawid, iProcessor, procTyp);

  //LogTrace("l1tOmtfEventPrint")<<"ADDING HIT: iLayer = " << iLayer << " iInput: " << iInput << " iPhi: " << iPhi << std::endl;

  //if (iLayer==17 && (iInput==0 || iInput==1)) continue;  // FIXME (MK) there is no RPC link for that input, because it is taken by DAQ link

  MuonStub stub;
  stub.type = MuonStub::RPC;
  stub.phiHw = angleConverter->getProcessorPhi(
      OMTFinputMaker::getProcessorPhiZero(config, iProcessor), procTyp, roll, cluster.firstStrip, cluster.lastStrip);
  stub.etaHw = angleConverter->getGlobalEtaRpc(rawid, cluster.firstStrip);

  stub.qualityHw = cluster.size();

  stub.bx = cluster.bx;
  stub.timing = cluster.timing;

  //stub.etaType = ?? TODO
  stub.logicLayer = iLayer;
  stub.detId = rawid;

  //This is very simple filtering of the clusters
  //Till Nov 2021: unfortunately performance of the firmware cannot be easily emulated from digi
  //(in principle would required raws, because in the firmware the clusterizaton is based on the 8-bit strip partitions)
  //The FW from from Nov 2021 solved this problem - option dropAllClustersIfMoreThanMax:
  //if any cluster is dropped in one barrel roll or endcap chamber - all are dropped for this roll/chamber.
  //Beside better data-to-emulator agreement it provides better eff for high pt muons
  if (config->getRpcDropAllClustersIfMoreThanMax()) {
    //two clusters were already added, so as we have the next one, we mark as dropped the one that was added before
    if (muonStubsInLayers[iLayer][iInput + 1]) {
      //if iInput+1 is not null, iInput is not null as well
      muonStubsInLayers[iLayer][iInput]->type = MuonStub::RPC_DROPPED;
      muonStubsInLayers[iLayer][iInput + 1]->type = MuonStub::RPC_DROPPED;
    } else if (cluster.size() > config->getRpcMaxClusterSize()) {
      //marking as dropped the one that was added before on the iInput
      if (muonStubsInLayers[iLayer][iInput]) {
        muonStubsInLayers[iLayer][iInput]->type = MuonStub::RPC_DROPPED;

        muonStubsInLayers[iLayer][iInput + 1] = std::make_shared<MuonStub>(stub);
        muonStubsInLayers[iLayer][iInput + 1]->type = MuonStub::RPC_DROPPED;
      } else {
        //no stub was added at this input already, so adding a stub and marking it as dropped
        muonStubsInLayers[iLayer].at(iInput) = std::make_shared<MuonStub>(stub);
        muonStubsInLayers[iLayer][iInput]->type = MuonStub::RPC_DROPPED;

        muonStubsInLayers[iLayer][iInput + 1] = std::make_shared<MuonStub>(stub);
        muonStubsInLayers[iLayer][iInput + 1]->type = MuonStub::RPC_DROPPED;
      }
    } else
      OMTFinputMaker::addStub(config, muonStubsInLayers, iLayer, iInput, stub);
  } else {
    if (cluster.size() <= config->getRpcMaxClusterSize())
      OMTFinputMaker::addStub(config, muonStubsInLayers, iLayer, iInput, stub);
  }

  std::ostringstream str;
  str << " RPC halfDigi "
      << " begin: " << cluster.firstStrip << " end: " << cluster.lastStrip << " iPhi: " << stub.phiHw
      << " iEta: " << stub.etaHw << " hwNumber: " << hwNumber << " iInput: " << iInput << " iLayer: " << iLayer
      << std::endl;

  LogTrace("l1tOmtfEventPrint") << str.str();
}

bool RpcDigiToStubsConverterOmtf::acceptDigi(const RPCDetId& rpcDetId, unsigned int iProcessor, l1t::tftype procType) {
  unsigned int aMin = config->getBarrelMin()[iProcessor];
  unsigned int aMax = config->getBarrelMax()[iProcessor];

  unsigned int aSector = rpcDetId.sector();

  ///Select RPC chambers connected to OMTF
  if (procType == l1t::tftype::omtf_pos &&
      (rpcDetId.region() < 0 || (rpcDetId.region() == 0 && rpcDetId.ring() != 2) ||
       (rpcDetId.region() == 0 && rpcDetId.station() == 4) ||
       (rpcDetId.region() == 0 && rpcDetId.station() == 2 && rpcDetId.layer() == 2 && rpcDetId.roll() == 1) ||
       (rpcDetId.region() == 0 && rpcDetId.station() == 3 && rpcDetId.roll() == 1) ||
       (rpcDetId.region() == 1 && rpcDetId.station() == 4) ||
       ///RPC RE1/2 temporarily not used (rpcDetId.region()==1 && rpcDetId.station()==1 && rpcDetId.ring()<2) ||
       (rpcDetId.region() == 1 && rpcDetId.station() > 0 && rpcDetId.ring() < 3)))
    return false;

  if (procType == l1t::tftype::omtf_neg &&
      (rpcDetId.region() > 0 || (rpcDetId.region() == 0 && rpcDetId.ring() != -2) ||
       (rpcDetId.region() == 0 && rpcDetId.station() == 4) ||
       (rpcDetId.region() == 0 && rpcDetId.station() == 2 && rpcDetId.layer() == 2 && rpcDetId.roll() == 1) ||
       (rpcDetId.region() == 0 && rpcDetId.station() == 3 && rpcDetId.roll() == 1) ||
       (rpcDetId.region() == -1 && rpcDetId.station() == 4) ||
       //RPC RE1/2 temporarily not used (rpcDetId.region()==1 && rpcDetId.station()==1 && rpcDetId.ring()<2) ||
       (rpcDetId.region() == -1 && rpcDetId.station() > 0 && rpcDetId.ring() < 3)))
    return false;

  if (procType == l1t::tftype::omtf_pos || procType == l1t::tftype::omtf_neg) {
    if (rpcDetId.region() != 0 && rpcDetId.station() == 3) {  //endcaps, layer 17
      unsigned int iInput = OMTFinputMaker::getInputNumber(config, rpcDetId.rawId(), iProcessor, procType);
      if (iInput == 0 || iInput == 1)
        return false;  // FIXME (MK) there is no RPC link for that input, because it is taken by DAQ link
    }
  }

  if (procType == l1t::tftype::bmtf && rpcDetId.region() != 0)
    return false;

  if (procType == l1t::tftype::emtf_pos &&
      (rpcDetId.region() <= 0 || (rpcDetId.station() == 1 && rpcDetId.ring() == 3)))
    return false;
  if (procType == l1t::tftype::emtf_neg &&
      (rpcDetId.region() >= 0 || (rpcDetId.station() == 1 && rpcDetId.ring() == 3)))
    return false;
  ////////////////
  if (rpcDetId.region() == 0)
    aSector = rpcDetId.sector();
  if (rpcDetId.region() != 0) {
    aSector = (rpcDetId.sector() - 1) * 6 + rpcDetId.subsector();
    aMin = config->getEndcap10DegMin()[iProcessor];
    aMax = config->getEndcap10DegMax()[iProcessor];
  }

  if (aMax > aMin && aSector >= aMin && aSector <= aMax)
    return true;
  if (aMax < aMin && (aSector >= aMin || aSector <= aMax))
    return true;

  return false;
}
///////////////////////////////////////
///////////////////////////////////////
bool OMTFinputMaker::acceptDtDigi(const OMTFConfiguration* config,
                                  const DTChamberId& dTChamberId,
                                  unsigned int iProcessor,
                                  l1t::tftype procType) {
  unsigned int aMin = config->getBarrelMin()[iProcessor];
  unsigned int aMax = config->getBarrelMax()[iProcessor];

  if (procType == l1t::tftype::omtf_pos && dTChamberId.wheel() != 2)
    return false;
  if (procType == l1t::tftype::omtf_neg && dTChamberId.wheel() != -2)
    return false;
  if (procType == l1t::tftype::emtf_pos || procType == l1t::tftype::emtf_neg)
    return false;

  unsigned int aSector = dTChamberId.sector();

  if (aMax > aMin && aSector >= aMin && aSector <= aMax)
    return true;
  if (aMax < aMin && (aSector >= aMin || aSector <= aMax))
    return true;

  return false;
}

///////////////////////////////////////
///////////////////////////////////////
OMTFinputMaker::OMTFinputMaker(const edm::ParameterSet& edmParameterSet,
                               MuStubsInputTokens& muStubsInputTokens,
                               const OMTFConfiguration* config,
                               std::unique_ptr<OmtfAngleConverter> angleConverter)
    : MuonStubMakerBase(config), config(config), angleConverter(std::move(angleConverter)) {
  if (!edmParameterSet.getParameter<bool>("dropDTPrimitives"))
    digiToStubsConverters.emplace_back(std::make_unique<DtDigiToStubsConverterOmtf>(
        config, this->angleConverter.get(), muStubsInputTokens.inputTokenDtPh, muStubsInputTokens.inputTokenDtTh));

  if (!edmParameterSet.getParameter<bool>("dropCSCPrimitives"))
    digiToStubsConverters.emplace_back(std::make_unique<CscDigiToStubsConverterOmtf>(
        config, this->angleConverter.get(), muStubsInputTokens.inputTokenCSC));

  if (!edmParameterSet.getParameter<bool>("dropRPCPrimitives"))
    digiToStubsConverters.emplace_back(std::make_unique<RpcDigiToStubsConverterOmtf>(
        config, this->angleConverter.get(), &rpcClusterization, muStubsInputTokens.inputTokenRPC));
}

void OMTFinputMaker::initialize(const edm::ParameterSet& edmCfg,
                                const edm::EventSetup& es,
                                const MuonGeometryTokens& muonGeometryTokens) {
  MuonStubMakerBase::initialize(edmCfg, es, muonGeometryTokens);
  angleConverter->checkAndUpdateGeometry(es, config, muonGeometryTokens);
}

///////////////////////////////////////
///////////////////////////////////////
OMTFinputMaker::~OMTFinputMaker() {}
///////////////////////////////////////
///////////////////////////////////////
unsigned int OMTFinputMaker::getInputNumber(const OMTFConfiguration* config,
                                            unsigned int rawId,
                                            unsigned int iProcessor,
                                            l1t::tftype type) {
  unsigned int iInput = 99;
  unsigned int aSector = 99;
  int aMin = config->getBarrelMin()[iProcessor];
  int iRoll = 1;
  int nInputsPerSector = 2;

  DetId detId(rawId);
  if (detId.det() != DetId::Muon)
    edm::LogError("Critical OMTFinputMaker") << "PROBLEM: hit in unknown Det, detID: " << detId.det() << std::endl;
  switch (detId.subdetId()) {
    case MuonSubdetId::RPC: {
      RPCDetId rpc(rawId);
      if (rpc.region() == 0) {
        nInputsPerSector = 4;
        aSector = rpc.sector();
        ///on the 0-2pi border we need to add 1 30 deg sector
        ///to get the correct index
        if (iProcessor == 5 && aSector < 3)
          aMin = -1;
        //Use division into rolls
        iRoll = rpc.roll();
        ///Set roll number by hand to keep common input
        ///number shift formula for all stations
        if (rpc.station() == 2 && rpc.layer() == 2 && rpc.roll() == 2)
          iRoll = 1;
        ///Only one roll from station 3 is connected.
        if (rpc.station() == 3) {
          iRoll = 1;
          nInputsPerSector = 2;
        }
        ///At the moment do not use RPC chambers splitting into rolls for bmtf part
        if (type == l1t::tftype::bmtf)
          iRoll = 1;
      }
      if (rpc.region() != 0) {
        aSector = (rpc.sector() - 1) * 6 + rpc.subsector();
        aMin = config->getEndcap10DegMin()[iProcessor];
        ///on the 0-2pi border we need to add 4 10 deg sectors
        ///to get the correct index
        if (iProcessor == 5 && aSector < 5)
          aMin = -4;
      }
      break;
    }
    case MuonSubdetId::DT: {
      DTChamberId dt(rawId);
      aSector = dt.sector();
      ///on the 0-2pi border we need to add 1 30 deg sector
      ///to get the correct index
      if (iProcessor == 5 && aSector < 3)
        aMin = -1;
      break;
    }
    case MuonSubdetId::CSC: {
      CSCDetId csc(rawId);
      aSector = csc.chamber();
      aMin = config->getEndcap10DegMin()[iProcessor];
      ///on the 0-2pi border we need to add 4 10deg sectors
      ///to get the correct index
      if (iProcessor == 5 && aSector < 5)
        aMin = -4;
      ///Endcap region covers algo 10 deg sectors
      ///on the 0-2pi border we need to add 2 20deg sectors
      ///to get the correct index
      if ((type == l1t::tftype::emtf_pos || type == l1t::tftype::emtf_neg) && csc.station() > 1 && csc.ring() == 1) {
        aMin = config->getEndcap20DegMin()[iProcessor];
        if (iProcessor == 5 && aSector < 3)
          aMin = -2;
      }
      break;
    }
  }

  ///Assume 2 hits per chamber
  iInput = (aSector - aMin) * nInputsPerSector;
  ///Chambers divided into two rolls have rolls number 1 and 3
  iInput += iRoll - 1;

  return iInput;
}
////////////////////////////////////////////
////////////////////////////////////////////

//iProcessor counted from 0
int OMTFinputMaker::getProcessorPhiZero(const OMTFConfiguration* config, unsigned int iProcessor) {
  unsigned int nPhiBins = config->nPhiBins();

  int phiZero = nPhiBins / 6. * (iProcessor) + nPhiBins / 24;
  // "0" is 15degree moved cyclically to each processor, note [0,2pi]

  return config->foldPhi(phiZero);
}

////////////////////////////////////////////
////////////////////////////////////////////
void OMTFinputMaker::addStub(const OMTFConfiguration* config,
                             MuonStubPtrs2D& muonStubsInLayers,
                             unsigned int iLayer,
                             unsigned int iInput,
                             MuonStub& stub) {
  //LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " iInput " << iInput << " " << stub << endl;
  //there is a small rate of duplicated digis in the real data in the DT and CSC, the reason for this duplicates is not understood
  //in case of RPC the duplicated digis appear in data only for the layer 17 (RE3), where the rolls 2 and 3 has the same eta = 115 assigned, and the muon can hit both rolls
  //the reason of the  duplicates cannot be the fact that the same data (links) goes to neighboring boards, because this is  removed in the OMTF unpacker
  //the duplicates cannot be dropped, because if they appear in data, are also duplicated on the input of the algorithm, and both can be used as the reference hits
  if (muonStubsInLayers[iLayer][iInput] && muonStubsInLayers[iLayer][iInput]->phiHw == stub.phiHw &&
      muonStubsInLayers[iLayer][iInput]->phiBHw == stub.phiBHw &&
      muonStubsInLayers[iLayer][iInput]->etaHw == stub.etaHw) {
    LogTrace("OMTFReconstruction") << "addStub: the stub with exactly the same phi, phiB and eta was already added:\n"
                                   << "incomnig stub " << stub << "\n"
                                   << "existing stub " << *(muonStubsInLayers[iLayer][iInput]) << std::endl;
    //return;
  }

  if (muonStubsInLayers[iLayer][iInput] && muonStubsInLayers[iLayer][iInput]->phiHw != (int)config->nPhiBins())
    ++iInput;

  if (muonStubsInLayers[iLayer][iInput] && muonStubsInLayers[iLayer][iInput]->phiHw != (int)config->nPhiBins())
    return;
  //in this implementation only two first stubs are added for a given iInput

  muonStubsInLayers.at(iLayer).at(iInput) = std::make_shared<MuonStub>(stub);
}
