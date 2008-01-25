#include "OnlineDB/CSCCondDB/interface/CSCDDUMapPopConAnalyzer.h"

using namespace popcon;

CSCDDUMapPopConAnalyzer::CSCDDUMapPopConAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<CSCDDUMap>(ps,"CSCDDUMap")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	std::cout << "Implemented CSCAnalyzer Constructor\n";
} 	

void CSCDDUMapPopConAnalyzer::CSCDDUMapPopConAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object =new CSCDDUMapImpl("CSCDDUMap",m_offline_connection,evt,est, m_pop_connection);
}

