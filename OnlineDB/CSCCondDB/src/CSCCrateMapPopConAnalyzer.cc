#include "OnlineDB/CSCCondDB/interface/CSCCrateMapPopConAnalyzer.h"

using namespace popcon;

CSCCrateMapPopConAnalyzer::CSCCrateMapPopConAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<CSCCrateMap>(ps,"CSCCrateMap")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	std::cout << "Implemented CSCAnalyzer Constructor\n";
} 	

void CSCCrateMapPopConAnalyzer::CSCCrateMapPopConAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object =new CSCCrateMapImpl("CSCCrateMap",m_offline_connection,evt,est, m_pop_connection);
}

