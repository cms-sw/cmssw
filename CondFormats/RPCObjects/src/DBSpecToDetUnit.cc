#include "CondFormats/RPCObjects/interface/DBSpecToDetUnit.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <string>

using namespace std;
using namespace edm;

uint32_t DBSpecToDetUnit::operator()(const ChamberLocationSpec & ch, 
   const FebLocationSpec & feb)
{
  static bool debug = edm::MessageDrop::instance()->debugEnabled;
  // REGION
  int region = -2;
  bool barrel = (ch.barrelOrEndcap==1);
  if (barrel) region = 0;
  else if (ch.diskOrWheel<0) region = -1;
  else if (ch.diskOrWheel>0) region = 1;

  //ROLL
  string localEtaPartVal[6]={"Forward","Central","Backward","A","B","C"};
  string nroll = localEtaPartVal[feb.localEtaPartition-1];

  //SUBSECTOR
  string subsecVal[5]={"--","-","0","+","++"};
  string subsec=subsecVal[ch.subsector+2];

  // build RPCdetId
  try {
    RPCDetId dn;
    dn.buildfromDB(region, ch.diskOrWheel, ch.layer, ch.sector, 
		   subsec, nroll, " ");
    return dn.rawId();
  } 
  catch(cms::Exception & e) {
    if (debug) LogDebug ("CondFormas/DBSpecToDetInit") 
      <<" Problem with RPCDetId, got exception!! " 
      <<"DB Chamber "<<ch.chamberLocationName()<<" roll "<<nroll
      <<e;
    return 0;
  }
}
