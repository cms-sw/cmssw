#ifndef RPC_RUN_IOV_H
#define RPC_RUN_IOV_H

/*
 * \class RPCRunIOV
 *  Reads data from ORCOFF and sqlite file
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondTools/RPC/interface/RPCDBSimSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/DataRecord/interface/RPCObCondRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"
#include "CoralBase/TimeStamp.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/RPC/interface/RPCFw.h"

class  RPCRunIOV {
 public:
  
  struct chRAW {
    int region;
    int ring;
    int station;
    int sector; 
    int layer; 
    int subsector;
  };
  
  RPCRunIOV();
  RPCRunIOV(const edm::EventSetup& evtSetup);
  ~RPCRunIOV();
  std::vector<RPCObImon::I_Item> getImon();
  std::vector<RPCObVmon::V_Item> getVmon();
  std::vector<RPCObTemp::T_Item> getTemp();
  std::map<int, RPCObPVSSmap::Item> getPVSSMap();
  bool isReadingNeeded(unsigned long long);
  unsigned long long toDAQ(unsigned long long);
  unsigned long long toUNIX(int, int);
  unsigned long long DAQtoUNIX(unsigned long long*);
  std::vector<RPCObImon::I_Item> filterIMON(std::vector<RPCObImon::I_Item>, unsigned long long, unsigned long long);
  std::string chamberName(chRAW);
  unsigned long long min_I;
  unsigned long long max_I;
  unsigned long long min_V;
  unsigned long long max_V;
  unsigned long long min_T;
  unsigned long long max_T;

private:
  const edm::EventSetup* eventSetup;
  std::vector<RPCObImon::I_Item> filtImon;
};

#endif
