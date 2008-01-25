#ifndef DDU_MAP_POPCON_IMPL_ANALYZER_H
#define DDU_MAP_POPCON_IMPL_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapHandler.h"


//
// class decleration
//

class CSCDDUMapPopConAnalyzer : public popcon::PopConAnalyzer<CSCDDUMap>
{
	public:
		CSCDDUMapPopConAnalyzer(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
