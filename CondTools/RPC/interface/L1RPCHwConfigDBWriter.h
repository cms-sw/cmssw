#ifndef L1RPCHWCONFIGDBWRITER
#define L1RPCHWCONFIGDBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RPC/interface/L1RPCHwConfigSourceHandler.h"


class L1RPCHwConfigDBWriter : public popcon::PopConAnalyzer<L1RPCHwConfig>
{
	public:
		L1RPCHwConfigDBWriter(const edm::ParameterSet&);
	private: 
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
                int m_validate;
};


#endif
