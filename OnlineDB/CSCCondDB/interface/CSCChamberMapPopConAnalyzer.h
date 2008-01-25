#ifndef CHAMBER_MAP_POPCON_IMPL_ANALYZER_H
#define CHAMBER_MAP_POPCON_IMPL_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapHandler.h"


//
// class decleration
//

class CSCChamberMapPopConAnalyzer : public popcon::PopConAnalyzer<CSCChamberMap>
{
	public:
		CSCChamberMapPopConAnalyzer(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
