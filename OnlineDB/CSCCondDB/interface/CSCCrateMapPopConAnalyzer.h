#ifndef CRATE_MAP_POPCON_IMPL_ANALYZER_H
#define CRATE_MAP_POPCON_IMPL_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapHandler.h"


//
// class decleration
//

class CSCCrateMapPopConAnalyzer : public popcon::PopConAnalyzer<CSCCrateMap>
{
	public:
		CSCCrateMapPopConAnalyzer(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
