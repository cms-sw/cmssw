//-------------------------------------------------
//
//   Class: RPCtoDTTranslator
//
//   RPCtoDTTranslator
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//--------------------------------------------------

#ifndef L1T_TwinMux_RPC_DTTranslator_H
#define L1T_TwinMux_RPC_DTTranslator_H

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "CondFormats/L1TObjects/interface/L1TTwinMuxParams.h"
#include "CondFormats/DataRecord/interface/L1TTwinMuxParamsRcd.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>

class RPCtoDTTranslator  {
public:
  RPCtoDTTranslator(RPCDigiCollection inrpcDigis);
  ~RPCtoDTTranslator() {};

  void run(const edm::EventSetup& c);

 ///Return Output PhContainer
 L1MuDTChambPhContainer getDTContainer(){  return m_rpcdt_translated;}
 L1MuDTChambPhContainer getDTRPCHitsContainer(){  return m_rpchitsdt_translated;}

 static int radialAngle(RPCDetId , const edm::EventSetup& , int);
 static int bendingAngle(int, int, int);
 //static int bendingAngle(int);
 static int localX(RPCDetId , const edm::EventSetup&, int );
 static int localXX(int, int, int );

private:

  ///Output PhContainer
  L1MuDTChambPhContainer m_rpcdt_translated;
  L1MuDTChambPhContainer m_rpchitsdt_translated;

  RPCDigiCollection m_rpcDigis;

  struct rpc_hit
  {
    int bx;
    int station;
    int sector;
    int wheel;
    RPCDetId detid;
    int strip;
    int roll;
    int layer;
    //rpc_hit(int pbx, int pstation,int psector, int pwheel, RPCDetId pdet, int pstrip, int proll, int player) : bx(pbx),station(pstation),sector(psector),wheel(pwheel, detid(pdet),strip(pstrip),roll(proll),layer(player) {}
  };


};
#endif
