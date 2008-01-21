#ifndef RPCREADOUTMAPPINGDBWRITER
#define RPCREADOUTMAPPINGDBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/RPCReadOutMappingSourceHandler.h"


class RPCReadOutMappingDBWriter : public popcon::PopConAnalyzer<RPCReadOutMapping>
{
	public:
		RPCReadOutMappingDBWriter(const edm::ParameterSet&);
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
