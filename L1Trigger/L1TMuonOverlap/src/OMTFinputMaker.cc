#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/AngleConverter.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

///////////////////////////////////////
///////////////////////////////////////
OMTFinputMaker::OMTFinputMaker() {}
///////////////////////////////////////
///////////////////////////////////////
void OMTFinputMaker::initialize(const edm::EventSetup& es, const OMTFConfiguration *omtfConfig){ 

  myAngleConverter.checkAndUpdateGeometry(es, omtfConfig->nPhiBins());

  myOmtfConfig = omtfConfig;
 
}
///////////////////////////////////////
///////////////////////////////////////
OMTFinputMaker::~OMTFinputMaker(){ }
///////////////////////////////////////
///////////////////////////////////////
bool  OMTFinputMaker::acceptDigi(uint32_t rawId,
				 unsigned int iProcessor,
				 l1t::tftype type){

  unsigned int aMin = myOmtfConfig->getBarrelMin()[iProcessor];
  unsigned int aMax = myOmtfConfig->getBarrelMax()[iProcessor];
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
      aMin = myOmtfConfig->getEndcap10DegMin()[iProcessor];
      aMax = myOmtfConfig->getEndcap10DegMax()[iProcessor];
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
    aMin = myOmtfConfig->getEndcap10DegMin()[iProcessor];
    aMax = myOmtfConfig->getEndcap10DegMax()[iProcessor];

    if( (type==l1t::tftype::emtf_pos || type==l1t::tftype::emtf_neg) &&
	csc.station()>1 && csc.ring()==1){
      aMin = myOmtfConfig->getEndcap20DegMin()[iProcessor];
      aMax = myOmtfConfig->getEndcap20DegMax()[iProcessor];
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
  int aMin = myOmtfConfig->getBarrelMin()[iProcessor];
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
      aMin = myOmtfConfig->getEndcap10DegMin()[iProcessor];
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
    aMin = myOmtfConfig->getEndcap10DegMin()[iProcessor];       
    ///on the 0-2pi border we need to add 4 10deg sectors
    ///to get the correct index
    if(iProcessor==5 && aSector<5) aMin = -4;
    ///Endcap region covers algo 10 deg sectors
    ///on the 0-2pi border we need to add 2 20deg sectors
    ///to get the correct index
    if( (type==l1t::tftype::emtf_pos || type==l1t::tftype::emtf_neg) &&
	csc.station()>1 && csc.ring()==1){
      aMin = myOmtfConfig->getEndcap20DegMin()[iProcessor];
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
OMTFinput OMTFinputMaker::processDT(const L1MuDTChambPhContainer *dtPhDigis,
	       const L1MuDTChambThContainer *dtThDigis,
	       unsigned int iProcessor,
	       l1t::tftype type, int bxTrg)
{

  OMTFinput result(myOmtfConfig);
  if(!dtPhDigis) return result;
  
  for (const auto digiIt: *dtPhDigis->getContainer()) {

    DTChamberId detid(digiIt.whNum(),digiIt.stNum(),digiIt.scNum()+1);

    ///Check it the data fits into given processor input range
    if(!acceptDigi(detid.rawId(), iProcessor, type)) continue;
    
    ///Check Trigger primitive quality
    ///Ts2Tag() == 0 - take only first track from DT Trigger Server
    ///BxCnt()  == 0 - ??
    ///code()>=3     - take only double layer hits, HH, HL and LL
    // FIXME (MK): at least Ts2Tag selection is not correct! Check it
//    if (digiIt.bxNum()!= 0 || digiIt.BxCnt()!= 0 || digiIt.Ts2Tag()!= 0 || digiIt.code()<4) continue;

    if (digiIt.bxNum()!= bxTrg) continue;
    
    if (myOmtfConfig->fwVersion() <= 4) {
      if (digiIt.code() != 4 && digiIt.code() != 5 && digiIt.code() != 6) continue;
    } else {
      if (digiIt.code() != 2 && digiIt.code() != 3 && digiIt.code() != 4 && digiIt.code() != 5 && digiIt.code() != 6) continue;
    }

    unsigned int hwNumber = myOmtfConfig->getLayerNumber(detid.rawId());
    if(myOmtfConfig->getHwToLogicLayer().find(hwNumber)==myOmtfConfig->getHwToLogicLayer().end()) continue;
    
    auto iter = myOmtfConfig->getHwToLogicLayer().find(hwNumber);
    unsigned int iLayer = iter->second;
    int iPhi =  myAngleConverter.getProcessorPhi(iProcessor, type, digiIt);
    int iEta =  myAngleConverter.getGlobalEta(detid.rawId(), digiIt, dtThDigis);
    unsigned int iInput= getInputNumber(detid.rawId(), iProcessor, type);    
    bool allowOverwrite = false;
    result.addLayerHit(iLayer,iInput,iPhi,iEta, allowOverwrite);
    result.addLayerHit(iLayer+1,iInput,digiIt.phiB(),iEta, allowOverwrite);    
  }

  return result;
  
}
////////////////////////////////////////////
////////////////////////////////////////////
OMTFinput OMTFinputMaker::processCSC(const CSCCorrelatedLCTDigiCollection *cscDigis,
	       unsigned int iProcessor,
	       l1t::tftype type, int bxTrg){

  OMTFinput result(myOmtfConfig);
  if(!cscDigis) return result;

  auto chamber = cscDigis->begin();
  auto chend  = cscDigis->end();
  for( ; chamber != chend; ++chamber ) {

    unsigned int rawid = (*chamber).first;
    ///Check it the data fits into given processor input range
    if(!acceptDigi(rawid, iProcessor, type)) continue;
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;    
    for( ; digi != dend; ++digi ) {

      ///Check if LCT trigger primitive has the right BX.
      if (digi->getBX()-CSCConstants::LCT_CENTRAL_BX != bxTrg) continue;

      unsigned int hwNumber = myOmtfConfig->getLayerNumber(rawid);
      if(myOmtfConfig->getHwToLogicLayer().find(hwNumber)==myOmtfConfig->getHwToLogicLayer().end()) continue;

      unsigned int iLayer = myOmtfConfig->getHwToLogicLayer().at(hwNumber);      
      int iPhi = myAngleConverter.getProcessorPhi(iProcessor, type, CSCDetId(rawid), *digi);
      int iEta = myAngleConverter.getGlobalEta(rawid, *digi);
      ///Accept CSC digis only up to eta=1.26.
      ///The nominal OMTF range is up to 1.24, but cutting at 1.24
      ///kill efficnency at the edge. 1.26 is one eta bin above nominal.
      //if(abs(iEta)>1.26/2.61*240) continue;
      //if (abs(iEta) > 115) continue;
      unsigned int iInput= getInputNumber(rawid, iProcessor, type);      
//    std::cout <<" ADDING CSC hit, proc: "<<iProcessor<<" iPhi : " << iPhi <<" iEta: "<< iEta << std::endl; 
      bool allowOverwrite = false;
      result.addLayerHit(iLayer,iInput,iPhi,iEta,allowOverwrite);     
    }
  }      
  return result;
}
////////////////////////////////////////////
////////////////////////////////////////////
bool rpcPrimitiveCmp(const  RPCDigi &a, const  RPCDigi &b) { return a.strip() < b.strip(); };
////////////////////////////////////////////
////////////////////////////////////////////
OMTFinput OMTFinputMaker::processRPC(const RPCDigiCollection *rpcDigis,
				unsigned int iProcessor,
				l1t::tftype type, int bxTrg){

  OMTFinput result(myOmtfConfig); 
  if(!rpcDigis) return result;
  std::stringstream str;

//  std::cout <<" RPC HITS, processor : " << iProcessor << std::endl;

  const RPCDigiCollection & rpcDigiCollection = *rpcDigis;
  for (auto rollDigis : rpcDigiCollection) {
    RPCDetId roll = rollDigis.first;    
    unsigned int rawid = roll.rawId();
    int nClusters = 0;
    if(!acceptDigi(rawid, iProcessor, type)) continue;    
    ///Find clusters of consecutive fired strips.
    ///Have to copy the digis in chamber to sort them (not optimal).
    ///NOTE: when copying I select only digis with bx==       //FIXME: find a better place/way to filtering digi against quality/BX etc.
//  for (auto tdigi = rollDigis.second.first; tdigi != rollDigis.second.second; tdigi++) { std::cout << "RPC DIGIS: " << roll.rawId()<< " "<<roll<<" digi: " << tdigi->strip() <<" bx: " << tdigi->bx() << std::endl; }
    std::vector<RPCDigi> digisCopy;
//  std::copy_if(rollDigis.second.first, rollDigis.second.second, std::back_inserter(digisCopy), [](const RPCDigi & aDigi){return (aDigi.bx()==0);} );
    for (auto pDigi=rollDigis.second.first; pDigi != rollDigis.second.second; pDigi++) { if (pDigi->bx()==bxTrg) digisCopy.push_back( *pDigi); }
    std::sort(digisCopy.begin(),digisCopy.end(),rpcPrimitiveCmp);
    typedef std::pair<unsigned int, unsigned int> Cluster;
    std::vector<Cluster> clusters;
    for(auto & digi: digisCopy) {
      if(clusters.empty()) clusters.push_back(Cluster(digi.strip(),digi.strip()));
      else if (digi.strip() - clusters.back().second == 1) clusters.back().second = digi.strip();
      else if (digi.strip() - clusters.back().second  > 1) clusters.push_back(Cluster(digi.strip(),digi.strip()));
    }

    for (auto & cluster: clusters) {
//      int iPhiHalfStrip1 = myAngleConverter.getProcessorPhi(iProcessor, type, roll, cluster.first);
//      int iPhiHalfStrip2 = myAngleConverter.getProcessorPhi(iProcessor, type, roll, cluster.second);
      int iPhi =  myAngleConverter.getProcessorPhi(iProcessor, type, roll, cluster.first, cluster.second);
      int cSize =  abs(int(cluster.first)-int(cluster.second))+1;
//      std::cout << " HStrip_1: " << iPhiHalfStrip1 <<" HStrip_2: "<<iPhiHalfStrip2<<" iPhi: " << iPhi << " cluster: ["<< cluster.first << ", "<<  cluster.second <<"]"<< std::endl;
      if (cSize>3) continue;
      int iEta =  myAngleConverter.getGlobalEta(rawid, cluster.first);      
      unsigned int hwNumber = myOmtfConfig->getLayerNumber(rawid);
      unsigned int iLayer = myOmtfConfig->getHwToLogicLayer().at(hwNumber);
      unsigned int iInput= getInputNumber(rawid, iProcessor, type);
//      std::cout <<"ADDING HIT: iLayer = " << iLayer << " iInput: " << iInput << " iPhi: " << iPhi << std::endl;
      if (iLayer==17 && (iInput==0 || iInput==1)) continue;  // FIXME (MK) there is no RPC link for that input, because it is taken by DAQ link
      bool outres = 
        result.addLayerHit(iLayer,iInput,iPhi,iEta);
//      if (cSize>2) flag |= 2;
//      if (!outres) flag |= 1;
      nClusters++;

      str <<" RPC halfDigi "
           <<" begin: "<<cluster.first<<" end: "<<cluster.second
           <<" iPhi: "<<iPhi
           <<" iEta: "<<iEta
           <<" hwNumber: "<<hwNumber
           <<" iInput: "<<iInput
           <<" iLayer: "<<iLayer
           <<" out: " << outres
           <<std::endl;      
    }
//    if (nClusters > 2) flag=1;
  }

  edm::LogInfo("OMTFInputMaker")<<str.str();
  return result;
}
////////////////////////////////////////////
////////////////////////////////////////////
OMTFinput OMTFinputMaker::buildInputForProcessor(const L1MuDTChambPhContainer *dtPhDigis,
							 const L1MuDTChambThContainer *dtThDigis,
							 const CSCCorrelatedLCTDigiCollection *cscDigis,
							 const RPCDigiCollection *rpcDigis,
							 unsigned int iProcessor,
							 l1t::tftype type,
                                           int bx){
  OMTFinput result(myOmtfConfig);
  result += processDT(dtPhDigis, dtThDigis, iProcessor, type, bx);
  result += processCSC(cscDigis, iProcessor, type, bx);
  result += processRPC(rpcDigis, iProcessor, type, bx);
  return result;
}
////////////////////////////////////////////
////////////////////////////////////////////
