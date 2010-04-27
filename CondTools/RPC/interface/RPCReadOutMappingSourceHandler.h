#ifndef RPCREADOUTMAPPINGSOURCEHANDLER
#define RPCREADOUTMAPPINGSOURCEHANDLER

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

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"
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
	class RPCReadOutMappingSourceHandler : public popcon::PopConSourceHandler<RPCReadOutMapping>
	{

		public:
    RPCReadOutMappingSourceHandler(const edm::ParameterSet& ps);
    ~RPCReadOutMappingSourceHandler();
    void getNewObjects();
    std::string id() const {return m_name;}
    void ConnectOnlineDB(string host, string sid, string user, string pass, int port);
    void DisconnectOnlineDB();
    void readCablingMap();
    int Compare2Cablings(const RPCReadOutMapping* map1, RPCReadOutMapping* map2);

		private:
    RPCReadOutMapping * cabling;
    Environment* env;
    Connection* conn;
    std::string m_name;
    int m_validate;
    std::string m_host;
    std::string m_sid;
    std::string m_user;
    std::string m_pass;
    int m_port;

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
