#ifndef L1RPCCONFIGSOURCEHANDLER
#define L1RPCCONFIGSOURCEHANDLER

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
//#include "CondCore/PopCon/interface/LogReader.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
#include <FWCore/ParameterSet/interface/FileInPath.h>
#include "L1Trigger/RPCTrigger/interface/RPCPatternsParser.h"

using namespace std;

namespace popcon
{
	class L1RPCConfigSourceHandler : public popcon::PopConSourceHandler<L1RPCConfig>
	{

		public:
    L1RPCConfigSourceHandler(const edm::ParameterSet& ps);
    ~L1RPCConfigSourceHandler();
    void getNewObjects();
    std::string id() const {return m_name;}
    void readConfig();
    int Compare2Configs(const L1RPCConfig* pat1, L1RPCConfig* pat2);

		private:
    L1RPCConfig * patterns;
    std::string m_name;
    int m_validate;
    int m_ppt;
    std::string m_dataDir;
    std::string m_patternsDir;

	};
}
#endif
