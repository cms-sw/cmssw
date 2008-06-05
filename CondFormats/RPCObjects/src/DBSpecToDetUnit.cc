#include "CondFormats/RPCObjects/interface/DBSpecToDetUnit.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <string>

using namespace std;
using namespace edm;

uint32_t DBSpecToDetUnit::operator()(const ChamberLocationSpec & ch, 
   const FebLocationSpec & feb)
{
  // REGION
  int region = -2;
  bool barrel = (ch.barrelOrEndcap=="Barrel");
  if (barrel) region = 0;
  else if (ch.diskOrWheel<0) region = -1;
  else if (ch.diskOrWheel>0) region = 1;

  //ROLL
  string nroll = feb.localEtaPartition;

  // build RPCdetId
  try {
    RPCDetId dn;
    dn.buildfromDB(region, ch.diskOrWheel, ch.layer, ch.sector, 
		   ch.subsector, nroll, ch.chamberLocationName);
    return dn.rawId();
  } 
  catch(...) {
    edm::LogWarning("CondFormas/DBSpecToDetInit") 
      <<" Problem with RPCDetId, got exception!! " 
      <<"DB Chamber "<<ch.chamberLocationName<<" roll "<<nroll;
    return 0;
  }
}
