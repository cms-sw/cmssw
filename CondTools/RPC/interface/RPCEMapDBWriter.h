#ifndef RPCEMAPDBWRITER
#define RPCEMAPDBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCEMapSourceHandler.h"


class RPCEMapDBWriter : public popcon::PopConAnalyzer<RPCEMap>
{
	public:
		RPCEMapDBWriter(const edm::ParameterSet&);
	private: 
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
                int m_validate;
                std::string m_host;
                std::string m_sid;
                std::string m_user;
                std::string m_pass;
                int m_port;
};


#endif
