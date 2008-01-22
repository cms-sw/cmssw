#ifndef PEDESTALSPOPCON_IMPL_ANALYZER_H
#define PEDESTLASPOPCON_IMPL_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCPedestalsHandler.h"


//
// class decleration
//

class CSCPedestalsPopConAnalyzer : public popcon::PopConAnalyzer<CSCDBPedestals>
{
	public:
		CSCPedestalsPopConAnalyzer(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
