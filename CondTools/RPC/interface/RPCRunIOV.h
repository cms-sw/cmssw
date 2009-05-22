#ifndef RPC_RUN_IOV_H
#define RPC_RUN_IOV_H

/*
 * \class RPCRunIOV
 *  Reads data from ORCOFF and sqlite file
 *
 *  $Date: 2009/05/20 11:16:49 $
 *  $Revision: 1.2 $
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
#include "CoralBase/TimeStamp.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/RPC/interface/RPCFw.h"

class  RPCRunIOV {
public:

  RPCRunIOV(unsigned long long m_since, unsigned long long m_till);
  RPCRunIOV(const edm::EventSetup& evtSetup);
  ~RPCRunIOV();
  std::vector<RPCObImon::I_Item> getImon();
  bool isReadingNeeded(unsigned long long);
  unsigned long long toDAQ(unsigned long long);
  unsigned long long toUNIX(int, int);
  unsigned long long min;
  unsigned long long max;

private:
  const edm::EventSetup* eventSetup;
  unsigned long long since;
  unsigned long long till;
  bool imon;
  bool print;
};

#endif
