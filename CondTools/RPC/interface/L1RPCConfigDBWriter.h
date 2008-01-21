#ifndef L1RPCCONFIGDBWRITER
#define L1RPCCONFIGDBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/L1RPCConfigSourceHandler.h"


class L1RPCConfigDBWriter : public popcon::PopConAnalyzer<L1RPCConfig>
{
	public:
		L1RPCConfigDBWriter(const edm::ParameterSet&);
	private: 
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
                int m_validate;
                int m_ppt;
                std::string m_dataDir;
};


#endif
