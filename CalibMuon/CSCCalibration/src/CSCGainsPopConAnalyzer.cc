#include "CalibMuon/CSCCalibration/interface/CSCGainsPopConAnalyzer.h"

using namespace popcon;

CSCGainsPopConAnalyzer::CSCGainsPopConAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<CSCDBGains>(ps,"CSCDBGains")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	std::cout << "Implemented CSCAnalyzer Constructor\n";
} 	

void CSCGainsPopConAnalyzer::CSCGainsPopConAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object =new CSCDBGainsImpl("CSCDBGains",m_offline_connection, m_catalog,evt,est, m_pop_connection);
}

