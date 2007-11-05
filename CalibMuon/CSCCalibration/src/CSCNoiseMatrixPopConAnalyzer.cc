#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixPopConAnalyzer.h"

using namespace popcon;

CSCNoiseMatrixPopConAnalyzer::CSCNoiseMatrixPopConAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<CSCDBNoiseMatrix>(ps,"CSCDBNoiseMatrix")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	std::cout << "Implemented CSCAnalyzer Constructor\n";
} 	

void CSCNoiseMatrixPopConAnalyzer::CSCNoiseMatrixPopConAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object =new CSCDBNoiseMatrixImpl("CSCDBNoiseMatrix",m_offline_connection, m_catalog,evt,est, m_pop_connection);
}

