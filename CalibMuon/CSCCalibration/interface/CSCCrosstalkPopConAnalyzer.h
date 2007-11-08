#ifndef CROSSTALKPOPCON_IMPL_ANALYZER_H
#define CROSSTALKPOPCON_IMPL_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCCrosstalkHandler.h"


//
// class decleration
//

class CSCCrosstalkPopConAnalyzer : public popcon::PopConAnalyzer<CSCDBCrosstalk>
{
	public:
		CSCCrosstalkPopConAnalyzer(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
