#ifndef GAINSPOPCON_IMPL_ANALYZER_H
#define GAINSPOPCON_IMPL_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCGainsHandler.h"


//
// class decleration
//

class CSCGainsPopConAnalyzer : public popcon::PopConAnalyzer<CSCDBGains>
{
	public:
		CSCGainsPopConAnalyzer(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
