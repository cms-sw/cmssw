#include <L1Trigger/L1TMuonBayes/interface/AngleConverterBase.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinput.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinputMaker.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

///////////////////////////////////////
///////////////////////////////////////
OMTFinputMaker::OMTFinputMaker(): rpcClusterization(3, 2) {

}

void OMTFinputMaker::initialize(const edm::ParameterSet& edmCfg, const edm::EventSetup& es, const OMTFConfiguration* procConf, MuStubsInputTokens& muStubsInputTokens) {
  this->config = procConf;
  MuonStubMakerBase::initialize(edmCfg, es, config, muStubsInputTokens);
  angleConverter.checkAndUpdateGeometry(es, config);
}

///////////////////////////////////////
///////////////////////////////////////
OMTFinputMaker::~OMTFinputMaker(){ }
///////////////////////////////////////
///////////////////////////////////////
bool  OMTFinputMaker::acceptDigi(uint32_t rawId,
    unsigned int iProcessor,
    l1t::tftype type){

  unsigned int aMin = config->getBarrelMin()[iProcessor];
  unsigned int aMax = config->getBarrelMax()[iProcessor];
  unsigned int aSector = 99;

  ///Clean up digis. Remove unconnected detectors
  DetId detId(rawId);
  if (detId.det() != DetId::Muon) 
    edm::LogError("Critical OMTFinputMaker") << "PROBLEM: hit in unknown Det, detID: "<<detId.det()<<std::endl;
  switch (detId.subdetId()) {
  case MuonSubdetId::RPC: {
    RPCDetId aId(rawId);

    ///Select RPC chambers connected to OMTF
    if(type==l1t::tftype::omtf_pos &&
        (aId.region()<0 ||
            (aId.region()==0 && aId.ring()!=2) ||
            (aId.region()==0 && aId.station()==4) ||
            (aId.region()==0 && aId.station()==2 && aId.layer()==2 && aId.roll()==1) ||
            (aId.region()==0 && aId.station()==3 && aId.roll()==1) ||
            (aId.region()==1 && aId.station()==4) ||
            ///RPC RE1/2 temporarily not used (aId.region()==1 && aId.station()==1 && aId.ring()<2) ||
            (aId.region()==1 && aId.station()>0 && aId.ring()<3))
    ) return false;

    if(type==l1t::tftype::omtf_neg &&
        (aId.region()>0 ||
            (aId.region()==0 && aId.ring()!=-2) ||
            (aId.region()==0 && aId.station()==4) ||
            (aId.region()==0 && aId.station()==2 && aId.layer()==2 && aId.roll()==1) ||
            (aId.region()==0 && aId.station()==3 && aId.roll()==1) ||
            (aId.region()==-1 && aId.station()==4) ||
            //RPC RE1/2 temporarily not used (aId.region()==1 && aId.station()==1 && aId.ring()<2) ||
            (aId.region()==-1 && aId.station()>0 && aId.ring()<3))
    ) return false;

    if(type==l1t::tftype::omtf_pos || type==l1t::tftype::omtf_neg) {
      if(aId.region()!=0 && aId.station() == 3) { //endcaps, layer 17
        unsigned int iInput= getInputNumber(rawId, iProcessor, type);
             if(iInput==0 || iInput==1)
               return false;  // FIXME (MK) there is no RPC link for that input, because it is taken by DAQ link
      }
    }

    if(type==l1t::tftype::bmtf && aId.region()!=0) return false;

    if(type==l1t::tftype::emtf_pos &&
        (aId.region()<=0 ||
            (aId.station()==1 && aId.ring()==3))) return false;
    if(type==l1t::tftype::emtf_neg &&
        (aId.region()>=0 ||
            (aId.station()==1 && aId.ring()==3))) return false;
    ////////////////
    if(aId.region()==0) aSector = aId.sector();
    if(aId.region()!=0){
      aSector = (aId.sector()-1)*6+aId.subsector();
      aMin = config->getEndcap10DegMin()[iProcessor];
      aMax = config->getEndcap10DegMax()[iProcessor];
    }

    break;
  }
  case MuonSubdetId::DT: {
    DTChamberId dt(rawId);

    if(type==l1t::tftype::omtf_pos && dt.wheel()!=2) return false;
    if(type==l1t::tftype::omtf_neg && dt.wheel()!=-2) return false;
    if(type==l1t::tftype::emtf_pos || type==l1t::tftype::emtf_neg) return false;

    aSector =  dt.sector();   	
    break;
  }
  case MuonSubdetId::CSC: {

    CSCDetId csc(rawId);    
    if(type==l1t::tftype::omtf_pos &&
        (csc.endcap()==2 || csc.ring()==1 || csc.station()==4)) return false;
    if(type==l1t::tftype::omtf_neg &&
        (csc.endcap()==1 || csc.ring()==1 || csc.station()==4)) return false;

    if(type==l1t::tftype::emtf_pos &&
        (csc.endcap()==2 || (csc.station()==1 && csc.ring()==3))
    ) return false;
    if(type==l1t::tftype::emtf_neg &&
        (csc.endcap()==1 || (csc.station()==1 && csc.ring()==3))
    ) return false;

    aSector =  csc.chamber();   	
    aMin = config->getEndcap10DegMin()[iProcessor];
    aMax = config->getEndcap10DegMax()[iProcessor];

    if( (type==l1t::tftype::emtf_pos || type==l1t::tftype::emtf_neg) &&
        csc.station()>1 && csc.ring()==1){
      aMin = config->getEndcap20DegMin()[iProcessor];
      aMax = config->getEndcap20DegMax()[iProcessor];
    }
    break;
  }    
  }

  if(aMax>aMin && aSector>=aMin && aSector<=aMax) return true;
  if(aMax<aMin && (aSector>=aMin || aSector<=aMax)) return true;

  return false;
}
///////////////////////////////////////
///////////////////////////////////////
unsigned int OMTFinputMaker::getInputNumber(unsigned int rawId, 
    unsigned int iProcessor,
    l1t::tftype type){

  unsigned int iInput = 99;
  unsigned int aSector = 99;
  int aMin = config->getBarrelMin()[iProcessor];
  int iRoll = 1;
  int nInputsPerSector = 2;

  DetId detId(rawId);
  if (detId.det() != DetId::Muon) edm::LogError("Critical OMTFinputMaker") << "PROBLEM: hit in unknown Det, detID: "<<detId.det()<<std::endl;
  switch (detId.subdetId()) {
  case MuonSubdetId::RPC: {
    RPCDetId rpc(rawId);
    if(rpc.region()==0){
      nInputsPerSector = 4;
      aSector = rpc.sector();
      ///on the 0-2pi border we need to add 1 30 deg sector
      ///to get the correct index
      if(iProcessor==5 && aSector<3) aMin = -1;
      //Use division into rolls
      iRoll = rpc.roll();
      ///Set roll number by hand to keep common input 
      ///number shift formula for all stations
      if(rpc.station()==2 && rpc.layer()==2 && rpc.roll()==2) iRoll = 1;
      ///Only one roll from station 3 is connected.
      if(rpc.station()==3){
        iRoll = 1;
        nInputsPerSector = 2;
      }
      ///At the moment do not use RPC chambers splitting into rolls for bmtf part      
      if(type==l1t::tftype::bmtf)iRoll = 1;
    }
    if(rpc.region()!=0){
      aSector = (rpc.sector()-1)*6+rpc.subsector();
      aMin = config->getEndcap10DegMin()[iProcessor];
      ///on the 0-2pi border we need to add 4 10 deg sectors
      ///to get the correct index
      if(iProcessor==5 && aSector<5) aMin = -4;
    }    
    break;
  }
  case MuonSubdetId::DT: {
    DTChamberId dt(rawId);
    aSector = dt.sector();    
    ///on the 0-2pi border we need to add 1 30 deg sector
    ///to get the correct index
    if(iProcessor==5 && aSector<3) aMin = -1;
    break;
  }
  case MuonSubdetId::CSC: {   
    CSCDetId csc(rawId);    
    aSector = csc.chamber();    
    aMin = config->getEndcap10DegMin()[iProcessor];       
    ///on the 0-2pi border we need to add 4 10deg sectors
    ///to get the correct index
    if(iProcessor==5 && aSector<5) aMin = -4;
    ///Endcap region covers algo 10 deg sectors
    ///on the 0-2pi border we need to add 2 20deg sectors
    ///to get the correct index
    if( (type==l1t::tftype::emtf_pos || type==l1t::tftype::emtf_neg) &&
        csc.station()>1 && csc.ring()==1){
      aMin = config->getEndcap20DegMin()[iProcessor];
      if(iProcessor==5 && aSector<3) aMin = -2;
    }
    break;
  }
  }

  ///Assume 2 hits per chamber
  iInput = (aSector - aMin)*nInputsPerSector;
  ///Chambers divided into two rolls have rolls number 1 and 3
  iInput+=iRoll-1;

  return iInput;
}
////////////////////////////////////////////
////////////////////////////////////////////

//iProcessor counted from 0
int OMTFinputMaker::getProcessorPhiZero(unsigned int iProcessor) {

  unsigned int nPhiBins = config->nPhiBins();

  int phiZero =  nPhiBins/6.*(iProcessor) + nPhiBins/24;  // "0" is 15degree moved cyclically to each processor, note [0,2pi]
  return config->foldPhi(phiZero);

}


void OMTFinputMaker::addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers, const L1MuDTChambPhDigi& digi,
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

  if (config->fwVersion() <= 4) {
    if (digi.code() != 4 && digi.code() != 5 && digi.code() != 6)
      return;
  } else {
    if (digi.code() != 2 && digi.code() != 3 && digi.code() != 4 && digi.code() != 5 && digi.code() != 6)
      return;
  }
/*  if (digi.code() != 4 && digi.code() != 5 && digi.code() != 6)
    return;*/
  //if (digiIt.code() != 2 && digiIt.code() != 3 && digiIt.code() != 4 && digiIt.code() != 5 && digiIt.code() != 6) continue;

  unsigned int hwNumber = config->getLayerNumber(detid.rawId());
  if(config->getHwToLogicLayer().find(hwNumber) == config->getHwToLogicLayer().end())
    return;

  auto iter = config->getHwToLogicLayer().find(hwNumber);
  unsigned int iLayer = iter->second;
  unsigned int iInput= getInputNumber(detid.rawId(), iProcessor, procTyp);
  //MuonStub& stub = muonStubsInLayers[iLayer][iInput];
  MuonStub stub;

  stub.type = MuonStub::DT_PHI_ETA;
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<" iProcessor "<<iProcessor<<std::endl;
  stub.phiHw  =  angleConverter.getProcessorPhi(getProcessorPhiZero(iProcessor), procTyp, digi);
  stub.etaHw  =  angleConverter.getGlobalEta(digi, dtThDigis);
  stub.phiBHw = digi.phiB();
  stub.qualityHw = digi.code();

  stub.bx = digi.bxNum(); //TODO sholdn't  it be BxCnt()?
  //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?

  //stub.etaType = ?? TODO
  stub.detId = detid;

  addStub(muonStubsInLayers, iLayer, iInput, stub);
}

////////////////////////////////////////////
////////////////////////////////////////////

void OMTFinputMaker::addCSCstubs(MuonStubPtrs2D& muonStubsInLayers, unsigned int rawid, const CSCCorrelatedLCTDigi& digi,
   unsigned int iProcessor, l1t::tftype procTyp)
{
  unsigned int hwNumber = config->getLayerNumber(rawid);
  if(config->getHwToLogicLayer().find(hwNumber) == config->getHwToLogicLayer().end())
    return;

  unsigned int iLayer = config->getHwToLogicLayer().at(hwNumber);
  unsigned int iInput= getInputNumber(rawid, iProcessor, procTyp);

  MuonStub stub;
  stub.type = MuonStub::CSC_PHI_ETA;
  stub.phiHw  =  angleConverter.getProcessorPhi(getProcessorPhiZero(iProcessor), procTyp, CSCDetId(rawid), digi);
  stub.etaHw  =  angleConverter.getGlobalEta(rawid, digi);
  stub.phiBHw = digi.getPattern(); //TODO change to phiB when implemented
  stub.qualityHw = digi.getQuality();

  stub.bx = digi.getBX(); //TODO sholdn't  it be getBX0()?
  //stub.timing = digi.getTiming(); //TODO what about sub-bx timing, is is available?

  //stub.etaType = ?? TODO
  stub.detId = rawid;

  addStub(muonStubsInLayers, iLayer, iInput, stub);
  ///Accept CSC digis only up to eta=1.26.
  ///The nominal OMTF range is up to 1.24, but cutting at 1.24
  ///kill efficnency at the edge. 1.26 is one eta bin above nominal.
  //if(abs(iEta)>1.26/2.61*240) continue;
  //if (abs(iEta) > 115) continue;

  //    std::cout <<" ADDING CSC hit, proc: "<<iProcessor<<" iPhi : " << iPhi <<" iEta: "<< iEta << std::endl;
}

////////////////////////////////////////////
////////////////////////////////////////////

void OMTFinputMaker::addRPCstub(MuonStubPtrs2D& muonStubsInLayers, const RPCDetId& roll, const RpcCluster& cluster,
   unsigned int iProcessor, l1t::tftype procTyp) {
  //      int iPhiHalfStrip1 = myangleConverter.getProcessorPhi(getProcessorPhiZero(iProcessor), type, roll, cluster.first);
  //      int iPhiHalfStrip2 = myangleConverter.getProcessorPhi(getProcessorPhiZero(iProcessor), type, roll, cluster.second);

  //unsigeint cSize =  cluster.size();

  //      std::cout << " HStrip_1: " << iPhiHalfStrip1 <<" HStrip_2: "<<iPhiHalfStrip2<<" iPhi: " << iPhi << " cluster: ["<< cluster.first << ", "<<  cluster.second <<"]"<< std::endl;
  //if (cSize>3) continue; this icut is allready in rpcClusterization.getClusters
  unsigned int rawid = roll.rawId();
  unsigned int hwNumber = config->getLayerNumber(rawid);
  unsigned int iLayer = config->getHwToLogicLayer().at(hwNumber);
  unsigned int iInput= getInputNumber(rawid, iProcessor, procTyp);
  //      std::cout <<"ADDING HIT: iLayer = " << iLayer << " iInput: " << iInput << " iPhi: " << iPhi << std::endl;
  //if (iLayer==17 && (iInput==0 || iInput==1)) continue;  // FIXME (MK) there is no RPC link for that input, because it is taken by DAQ link

  MuonStub stub;
  stub.type = MuonStub::RPC;
  stub.phiHw  =  angleConverter.getProcessorPhi(getProcessorPhiZero(iProcessor), procTyp, roll, cluster.firstStrip, cluster.lastStrip);
  stub.etaHw  =  angleConverter.getGlobalEta(rawid, cluster.firstStrip);
  angleConverter.AngleConverterBase::getGlobalEta(rawid, cluster.firstStrip);
  //stub.phiBHw =
  stub.qualityHw = cluster.size();

  stub.bx = cluster.bx;
  stub.timing = cluster.timing;

  //stub.etaType = ?? TODO
  stub.detId = rawid;

  addStub(muonStubsInLayers, iLayer, iInput, stub);

  //      if (cSize>2) flag |= 2;
  //      if (!outres) flag |= 1;

  std::ostringstream str;
  str <<" RPC halfDigi "
      <<" begin: "<<cluster.firstStrip<<" end: "<<cluster.lastStrip
      <<" iPhi: "<<stub.phiHw
      <<" iEta: "<<stub.etaHw
      <<" hwNumber: "<<hwNumber
      <<" iInput: "<<iInput
      <<" iLayer: "<<iLayer
      //<<" out: " << outres
      <<std::endl;

  edm::LogInfo("MuonStubMaker")<<str.str();
}

////////////////////////////////////////////
////////////////////////////////////////////
void OMTFinputMaker::addStub(MuonStubPtrs2D& muonStubsInLayers, unsigned int iLayer, unsigned int iInput, MuonStub& stub) {

  //in principle it is possible that in the DAQ data the digis are duplicated,
  //since the same link is connected to two OMTF boards
  //in principle this dupliactes should be already reoomved in the OMTF uncpacer, but just in case...
  if( muonStubsInLayers[iLayer][iInput] &&
      muonStubsInLayers[iLayer][iInput]->phiHw == stub.phiHw &&
      muonStubsInLayers[iLayer][iInput]->phiBHw == stub.phiBHw &&
      muonStubsInLayers[iLayer][iInput]->etaHw == stub.etaHw) {
    edm::LogWarning("OMTFInputMaker")<<"addStub: the stub with exactly the same phi, phiB and eta was already added, stub.type: "<<stub.type;
    return;
  }

  if(muonStubsInLayers[iLayer][iInput] && muonStubsInLayers[iLayer][iInput]->phiHw != (int)config->nPhiBins())
    ++iInput;

  if(muonStubsInLayers[iLayer][iInput] && muonStubsInLayers[iLayer][iInput]->phiHw != (int)config->nPhiBins())
    return;
  //in this implementation only two first stubs are added for a given iInput

  muonStubsInLayers.at(iLayer).at(iInput) = std::make_shared<const MuonStub>(stub);
  //cout<<__FUNCTION__<<":"<<__LINE__<<" stub phi "<<stub.phiHw<<endl;
}

////////////////////////////////////////////
////////////////////////////////////////////
/*const OMTFinput OMTFinputMaker::buildInputForProcessor(const L1MuDTChambPhContainer *dtPhDigis,
    const L1MuDTChambThContainer *dtThDigis,
    const CSCCorrelatedLCTDigiCollection *cscDigis,
    const RPCDigiCollection *rpcDigis,
    unsigned int iProcessor,
    l1t::tftype type,
    int bx) {
  OMTFinput result(config);
  //MuonStubPtrs2D& muonStubsInLayers = result.getMuonStubs();
  int bxFrom = bx, bxTo = bx;
  processDT(result.getMuonStubs(), dtPhDigis, dtThDigis, iProcessor, type, true, bxFrom, bxTo);
  processCSC(result.getMuonStubs(), cscDigis, iProcessor, type, bxFrom, bxTo);
  processRPC(result.getMuonStubs(), rpcDigis, iProcessor, type, bxFrom, bxTo);
  //cout<<result<<endl;
  return result;
}*/
////////////////////////////////////////////
////////////////////////////////////////////
