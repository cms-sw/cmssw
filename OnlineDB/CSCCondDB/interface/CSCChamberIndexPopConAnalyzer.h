#ifndef CHAMBER_INDEXPOPCON_IMPL_ANALYZER_H
#define CHAMBER_INDEXPOPCON_IMPL_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexHandler.h"


//
// class decleration
//

class CSCChamberIndexPopConAnalyzer : public popcon::PopConAnalyzer<CSCChamberIndex>
{
	public:
		CSCChamberIndexPopConAnalyzer(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
