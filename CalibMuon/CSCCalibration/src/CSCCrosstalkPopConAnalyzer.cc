#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkPopConAnalyzer.h"

using namespace popcon;

CSCCrosstalkPopConAnalyzer::CSCCrosstalkPopConAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<CSCDBCrosstalk>(ps,"CSCDBCrosstalk")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	std::cout << "Implemented CSCAnalyzer Constructor\n";
} 	

void CSCCrosstalkPopConAnalyzer::CSCCrosstalkPopConAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object =new CSCDBCrosstalkImpl("CSCDBCrosstalk",m_offline_connection, m_catalog,evt,est, m_pop_connection);
}

