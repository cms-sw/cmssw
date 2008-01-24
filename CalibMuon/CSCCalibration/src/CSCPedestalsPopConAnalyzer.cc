#include "CalibMuon/CSCCalibration/interface/CSCPedestalsPopConAnalyzer.h"

using namespace popcon;

CSCPedestalsPopConAnalyzer::CSCPedestalsPopConAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<CSCDBPedestals>(ps,"CSCDBPedestals")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	std::cout << "Implemented CSCAnalyzer Constructor\n";
} 	

void CSCPedestalsPopConAnalyzer::CSCPedestalsPopConAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object =new CSCDBPedestalsImpl("CSCDBPedestals",m_offline_connection,evt,est, m_pop_connection);
}

