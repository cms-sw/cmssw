#ifndef L1RPCCONFIGSOURCEHANDLER
#define L1RPCCONFIGSOURCEHANDLER

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/PopCon/interface/LogReader.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/RPCObjects/interface/L1RPCConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
#include <FWCore/ParameterSet/interface/FileInPath.h>
#include "L1Trigger/RPCTrigger/interface/RPCPatternsParser.h"

using namespace std;

namespace popcon
{
	class L1RPCConfigSourceHandler : public popcon::PopConSourceHandler<L1RPCConfig>
	{

		public:
    L1RPCConfigSourceHandler(const std::string&, const std::string&, const edm::Event& evt, const edm::EventSetup& est, int validate, int ppt, std::string dataDir);
    ~L1RPCConfigSourceHandler();
    void getNewObjects();
    void readConfig();
    int Compare2Configs(const L1RPCConfig* pat1, L1RPCConfig* pat2);

		private:
    L1RPCConfig * patterns;
    int m_validate;
    int m_ppt;
    std::string m_dataDir;
    std::string m_patternsDir;

	};
}
#endif
