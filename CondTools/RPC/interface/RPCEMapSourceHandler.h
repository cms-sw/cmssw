#ifndef RPCEMAPSOURCEHANDLER
#define RPCEMAPSOURCEHANDLER

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "OnlineDB/Oracle/interface/Oracle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
//#include "CondCore/PopCon/interface/LogReader.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "CondFormats/RPCObjects/interface/DBSpecToDetUnit.h"

using namespace std;
using namespace oracle::occi;

namespace popcon
{
	class RPCEMapSourceHandler : public popcon::PopConSourceHandler<RPCEMap>
	{

		public:
    RPCEMapSourceHandler(const edm::ParameterSet& ps);
    ~RPCEMapSourceHandler();
    void getNewObjects();
    std::string id() const {return m_name;}
    void ConnectOnlineDB(string host, string sid, string user, string pass, int port);
    void DisconnectOnlineDB();
    void readEMap();
    int Compare2EMaps(const RPCEMap* map1, RPCEMap* map2);

		private:
    RPCEMap * eMap;
    Environment* env;
    Connection* conn;
    std::string m_name;
    int m_validate;
    std::string m_host;
    std::string m_sid;
    std::string m_user;
    std::string m_pass;
    int m_port;
//    const edm::Event& event; 	 
//    const edm::EventSetup& esetup;

  // utilities
    string IntToString(int num)
    {
      stringstream snum;
      snum << num << flush;
      return(snum.str());
    }

    typedef struct{int febId,chamberId,connectorId,lbInputNum,posInLocalEtaPart,posInCmsEtaPart;string localEtaPart,cmsEtaPart;} FEBStruct;
	};
}
#endif
