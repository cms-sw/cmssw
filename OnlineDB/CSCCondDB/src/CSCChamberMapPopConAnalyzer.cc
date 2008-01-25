#include "OnlineDB/CSCCondDB/interface/CSCChamberMapPopConAnalyzer.h"

using namespace popcon;

CSCChamberMapPopConAnalyzer::CSCChamberMapPopConAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<CSCChamberMap>(ps,"CSCChamberMap")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	std::cout << "Implemented CSCAnalyzer Constructor\n";
} 	

void CSCChamberMapPopConAnalyzer::CSCChamberMapPopConAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object =new CSCChamberMapImpl("CSCChamberMap",m_offline_connection,evt,est, m_pop_connection);
}

