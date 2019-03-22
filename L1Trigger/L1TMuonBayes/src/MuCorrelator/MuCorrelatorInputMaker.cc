/*
 * MuonCorrelatorInputMaker.cc
 *
 *  Created on: Jan 30, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include <L1Trigger/L1TMuonBayes/interface/MuCorrelator/MuCorrelatorInputMaker.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MuCorrelatorInputMaker::MuCorrelatorInputMaker(const edm::ParameterSet& edmCfg, const edm::EventSetup& es, MuCorrelatorConfigPtr config, MuStubsInputTokens muStubsInputTokens):
  config(config), rpcClusterization(3, 2), muStubsInputTokens(muStubsInputTokens)
{
  angleConverter.checkAndUpdateGeometry(es, config.get() );

  dropDTPrimitives = edmCfg.getParameter<bool>("dropDTPrimitives");
  dropRPCPrimitives = edmCfg.getParameter<bool>("dropRPCPrimitives");
  dropCSCPrimitives = edmCfg.getParameter<bool>("dropCSCPrimitives");

  if(edmCfg.exists("minDtPhQuality") ) {
    minDtPhQuality = edmCfg.getParameter<int>("minDtPhQuality");
  }
}

MuCorrelatorInputMaker::~MuCorrelatorInputMaker() {
  // TODO Auto-generated destructor stub
}


void MuCorrelatorInputMaker::addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers, const L1MuDTChambPhDigi& digi,
    const L1MuDTChambThContainer *dtThDigis,
    unsigned int iProcessor, l1t::tftype procTyp)
{

  DTChamberId detid(digi.whNum(), digi.stNum(), digi.scNum()+1);

  ///Check Trigger primitive quality
  ///Ts2Tag() == 0 - take only first track from DT Trigger Server
  ///BxCnt()  == 0 - ??
  ///code()>=3     - take only double layer hits, HH, HL and LL
  // FIXME (MK): at least Ts2Tag selection is not correct! Check it
  //    if (digiIt.bxNum()!= 0 || digiIt.BxCnt()!= 0 || digiIt.Ts2Tag()!= 0 || digiIt.code()<4) continue;

  //if (digi.code() != 4 && digi.code() != 5 && digi.code() != 6) return; //TODO onluy for the pdf generation
  //if (digi.code() != 2 && digi.code() != 3 && digi.code() != 4 && digi.code() != 5 && digi.code() != 6) return;
  if(digi.code() ==  7 || digi.code() < minDtPhQuality) //7 is empty digi, TODO update if the definition of the quality is changed
    return;

  unsigned int iLayer = getLayerNumber(detid);

  MuonStub stub;

  stub.type = MuonStub::DT_PHI;
  stub.phiHw  =  angleConverter.getProcessorPhi(0, procTyp, digi);

  EtaValue etaVal = angleConverter.getGlobalEtaDt(detid);
  stub.etaHw  = etaVal.eta;
  stub.etaSigmaHw = etaVal.etaSigma;

  stub.phiBHw = digi.phiB();
  stub.qualityHw = digi.code();

  stub.bx = digi.bxNum(); //TODO sholdn't  it be BxCnt()?
  //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?
  stub.timing = digi.bxNum() * 2; //TODO temporary solution untill the real sub-bx timing is provided

  stub.roll = abs(digi.whNum());
  //if(roll.ring() == 0 && roll.sector() %2) {//in wheel zero in the odd sectors the chambers are placed from the other side than in even, thus the rolls have to be swapped
    //not so easy - the cells are also readout from the other side. separate rolls are needed

  //stub.etaType = ?? TODO

  stub.logicLayer = iLayer;
  stub.detId = detid;

  addStub(muonStubsInLayers, iLayer, stub);
}


void MuCorrelatorInputMaker::addDTetaStubs(MuonStubPtrs2D& muonStubsInLayers, const L1MuDTChambThDigi& thetaDigi,
    unsigned int iProcessor, l1t::tftype procTyp)
{
  DTChamberId detid(thetaDigi.whNum(), thetaDigi.stNum(), thetaDigi.scNum()+1);

  unsigned int iLayer = getLayerNumber(detid, true);

  std::vector<EtaValue> etaSegments;
  angleConverter.getGlobalEta(thetaDigi, etaSegments);

  for(auto& etaVal : etaSegments) {
    MuonStub stub;

    stub.type = MuonStub::DT_THETA;
    //stub.phiHw  =  angleConverter.getProcessorPhi(0, procTyp, thetaDigi); TODO implement

    stub.etaHw  = etaVal.eta;
    stub.etaSigmaHw = etaVal.etaSigma;

    stub.phiBHw = 0;
    stub.qualityHw = etaVal.quality;

    stub.bx = etaVal.bx; //TODO sholdn't  it be BxCnt()?
    //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?

    //stub.etaType = ?? TODO

    stub.logicLayer = iLayer;
    stub.detId = detid;

    addStub(muonStubsInLayers, iLayer, stub);
  }
}

////////////////////////////////////////////
////////////////////////////////////////////

void MuCorrelatorInputMaker::addCSCstubs(MuonStubPtrs2D& muonStubsInLayers, unsigned int rawid, const CSCCorrelatedLCTDigi& digi,
   unsigned int iProcessor, l1t::tftype procTyp)
{

  CSCDetId detId(rawid);
  {
    unsigned int iLayer = getLayerNumber(detId);

    MuonStub stub;
    stub.type = MuonStub::CSC_PHI;
    stub.phiHw  =  angleConverter.getProcessorPhi(0, procTyp, detId, digi);

    EtaValue etaVal = angleConverter.getGlobalEtaCsc(rawid); //middle of the chamber
    stub.etaHw  = etaVal.eta;
    stub.etaSigmaHw = etaVal.etaSigma;

    stub.phiBHw = digi.getPattern(); //TODO change to phiB when implemented
    stub.qualityHw = digi.getQuality();

    stub.bx = digi.getBX(); //TODO sholdn't  it be getBX0()?
    //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?
    stub.timing = digi.getBX() - 6; //TODO move 6 to config

    //stub.etaType = ?? TODO

    stub.logicLayer = iLayer;
    stub.detId = rawid;

    if(detId.ring() == 4) //ME1/1 a, for //ME1/1 there is separate layer, so it should be ok move it to roll 1
      stub.roll = 1;
    else
      stub.roll = abs(detId.ring()-1);

    //stub.phi = config->getProcScalePhiToRad(stub.phiHw);
    //stub.eta = config->hwEtaToEta(stub.etaHw);

    addStub(muonStubsInLayers, iLayer, stub);

    //cout<<__FUNCTION__<<":"<<__LINE__<<" adding CSC phi stub from chamber "<<detId<<" "<<stub<<endl;
  }

  { //adding eta stub
    unsigned int iLayer = getLayerNumber(detId, true);

    MuonStub stub;
    stub.type = MuonStub::CSC_ETA;
    //stub.phiHw  =  angleConverter.getProcessorPhi(0, procTyp, detId, digi);

    EtaValue etaVal = angleConverter.getGlobalEta(rawid, digi); //eta of the segment, the separate phi and eta segments are added
    stub.etaHw  = etaVal.eta;
    stub.etaSigmaHw = etaVal.etaSigma;

    stub.qualityHw = digi.getQuality();

    stub.bx = digi.getBX() * 2; //TODO sholdn't  it be getBX0()?
    //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?

    //stub.etaType = ?? TODO

    stub.logicLayer = iLayer;
    stub.detId = rawid;

    //stub.phi = config->getProcScalePhiToRad(stub.phiHw);
    //stub.eta = config->hwEtaToEta(stub.etaHw);

    addStub(muonStubsInLayers, iLayer, stub);

    //cout<<__FUNCTION__<<":"<<__LINE__<<" adding CSC eta stub from chamber "<<detId<<" "<<stub<<endl;
  }

}

////////////////////////////////////////////
////////////////////////////////////////////

void MuCorrelatorInputMaker::addRPCstub(MuonStubPtrs2D& muonStubsInLayers, const RPCDetId& roll, const RpcCluster& cluster,
   unsigned int iProcessor, l1t::tftype procTyp) {
  //      int iPhiHalfStrip1 = myangleConverter.getProcessorPhi(getProcessorPhiZero(iProcessor), type, roll, cluster.first);
  //      int iPhiHalfStrip2 = myangleConverter.getProcessorPhi(getProcessorPhiZero(iProcessor), type, roll, cluster.second);

  //unsigeint cSize =  cluster.size();

  //      std::cout << " HStrip_1: " << iPhiHalfStrip1 <<" HStrip_2: "<<iPhiHalfStrip2<<" iPhi: " << iPhi << " cluster: ["<< cluster.first << ", "<<  cluster.second <<"]"<< std::endl;
  //if (cSize>3) continue; this icut is allready in rpcClusterization.getClusters
  unsigned int rawid = roll.rawId();
  //      std::cout <<"ADDING HIT: iLayer = " << iLayer << " iInput: " << iInput << " iPhi: " << iPhi << std::endl;
  //if (iLayer==17 && (iInput==0 || iInput==1)) continue;  // FIXME (MK) there is no RPC link for that input, because it is taken by DAQ link

  unsigned int iLayer = getLayerNumber(roll);

  MuonStub stub;
  stub.type = MuonStub::RPC;
  stub.phiHw  =  angleConverter.getProcessorPhi(0, procTyp, roll, cluster.firstStrip, cluster.lastStrip);

  EtaValue etaVal = angleConverter.getGlobalEta(rawid, cluster.firstStrip);
  stub.etaHw  = etaVal.eta;
  stub.etaSigmaHw = etaVal.etaSigma;

  angleConverter.AngleConverterBase::getGlobalEta(rawid, cluster.firstStrip);
  //stub.phiBHw =
  stub.qualityHw = cluster.size();
  //stub.etaType = ?? TODO

  stub.bx = cluster.bx;
  stub.timing = cluster.timing;

  stub.logicLayer = iLayer;

  if(roll.region() == 0) {//barel
    stub.roll = abs(roll.ring()) * 3 + roll.roll()-1;
    if(roll.ring() == 0 && roll.sector() %2) {//in wheel zero in the odd sectors the chambers are placed from the other side than in even, thus the rolls have to be swapped
      //not so easy - the strips are also readout from the other side. separate rolls are needed
      if(stub.roll == 0)
        stub.roll = 2;
      else if(stub.roll == 1)
        stub.roll = 0;
    }
  }
  else {
    stub.roll = (roll.ring() -1) * 3 + roll.roll()-1;
  }

  stub.detId = rawid;

  addStub(muonStubsInLayers, iLayer, stub);

  //      if (cSize>2) flag |= 2;
  //      if (!outres) flag |= 1;

/*  std::ostringstream str;
  str <<" RPC halfDigi "
      <<" begin: "<<cluster.firstStrip<<" end: "<<cluster.lastStrip
      <<" iPhi: "<<stub.phiHw
      <<" iEta: "<<stub.etaHw
      <<" iLayer: "<<iLayer
      //<<" out: " << outres
      <<std::endl;

  edm::LogInfo("MuonStubMaker")<<str.str();*/
}

////////////////////////////////////////////
////////////////////////////////////////////
void MuCorrelatorInputMaker::addStub(MuonStubPtrs2D& muonStubsInLayers, unsigned int iLayer, MuonStub& stub) {
  //in principle it is possible that in the DAQ data the digis are duplicated,
  //since the same link is connected to two OMTF boards
  //in principle this dupliactes should be already reoomved in the OMTF uncpacer, but just in case...
/*  if( muonStubsInLayers[iLayer][iInput] &&
      muonStubsInLayers[iLayer][iInput]->phiHw == stub.phiHw &&
      muonStubsInLayers[iLayer][iInput]->phiBHw == stub.phiBHw &&
      muonStubsInLayers[iLayer][iInput]->etaHw == stub.etaHw) {
    edm::LogWarning("OMTFInputMaker")<<"addStub: the stub with exactly the same phi, phiB and eta was already added, stub.type: "<<stub.type;
    return;
  }*/ //todo add checking if there is the same stub alrady?

  if(muonStubsInLayers.at(iLayer).size() < config->nMaxMuStubsPerLayer() )
    muonStubsInLayers.at(iLayer).emplace_back(std::make_shared<const MuonStub>(stub));

  //cout<<__FUNCTION__<<":"<<__LINE__<<" stub phi "<<stub.phiHw<<endl;
}

uint32_t MuCorrelatorInputMaker::getLayerNumber(const DTChamberId& detid, bool eta) const {
  //station is counted from 1
  if(eta)
    return (detid.station() -1) + config->nPhiLayers();

  return (detid.station() -1) * 2;
}

uint32_t MuCorrelatorInputMaker::getLayerNumber(const CSCDetId& detid, bool eta) const {
  if(eta)
    return (detid.station() -1)+ 3 + config->nPhiLayers() ;

  if(detid.station() == 1 && (detid.ring() ==  1 || detid.ring() == 4) ){ //ME1/1 phi
    return 8;
  }

  return (detid.station() -1) + 8 + 1; //8 is DT layers number - two per station + ME1/1
}

uint32_t MuCorrelatorInputMaker::getLayerNumber(const RPCDetId& detid) const {
  //TODO configure somehow from config?
  if(detid.region() == 0) { //barrel
    uint32_t rpcLogLayer = 0;
    if(detid.station() == 1)
      rpcLogLayer = 0 + detid.layer() -1;
    else if(detid.station() == 2)
      rpcLogLayer = 2 + detid.layer() -1;
    else //station 3 and 4
      rpcLogLayer = detid.station() + 1;

    //cout<<__FUNCTION__<<":"<<__LINE__<<" RPC detid "<<detid<<" rpcLogLayer "<<rpcLogLayer<<endl;
    return (rpcLogLayer + 8 + 5); //8 is DT layers number - two per station, 5 is CSC layer number
  }
  //endcap
  return (detid.station() -1) + 8 + 5 + 6; //8 is DT layers number - two per station, 5 is CSC layer number, 6 is RPC barrel station number
}




void MuCorrelatorInputMaker::loadAndFilterDigis(const edm::Event& event) {
  // Filter digis by dropping digis from selected (by cfg.py) subsystems
  if(!dropDTPrimitives){
    event.getByToken(muStubsInputTokens.inputTokenDTPh, dtPhDigis);
    event.getByToken(muStubsInputTokens.inputTokenDTTh, dtThDigis);
  }
  if(!dropRPCPrimitives) event.getByToken(muStubsInputTokens.inputTokenRPC, rpcDigis);
  if(!dropCSCPrimitives) event.getByToken(muStubsInputTokens.inputTokenCSC, cscDigis);
}




const MuonStubsInput MuCorrelatorInputMaker::buildInputForProcessor(unsigned int iProcessor,
    l1t::tftype type,
    int bxFrom, int bxTo) {

  MuonStubsInput result(config);
  processDT( result.getMuonStubs(), dtPhDigis.product(), dtThDigis.product(), iProcessor, type, false, bxFrom, bxTo);
  processCSC(result.getMuonStubs(), cscDigis.product(), iProcessor, type, bxFrom, bxTo);
  processRPC(result.getMuonStubs(), rpcDigis.product(), iProcessor, type, bxFrom, bxTo);
  //cout<<result<<endl;
  return result;
}
