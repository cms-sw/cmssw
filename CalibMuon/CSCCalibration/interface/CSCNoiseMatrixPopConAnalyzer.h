#ifndef NOISEMATRIXPOPCON_IMPL_ANALYZER_H
#define NOISEMATRIXPOPCON_IMPL_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCNoiseMatrixHandler.h"


//
// class decleration
//

class CSCNoiseMatrixPopConAnalyzer : public popcon::PopConAnalyzer<CSCDBNoiseMatrix>
{
	public:
		CSCNoiseMatrixPopConAnalyzer(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
