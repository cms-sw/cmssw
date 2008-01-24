#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexPopConAnalyzer.h"

using namespace popcon;

CSCChamberIndexPopConAnalyzer::CSCChamberIndexPopConAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<CSCChamberIndex>(ps,"CSCChamberIndex")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	std::cout << "Implemented CSCAnalyzer Constructor\n";
} 	

void CSCChamberIndexPopConAnalyzer::CSCChamberIndexPopConAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object =new CSCChamberIndexImpl("CSCChamberIndex",m_offline_connection,evt,est, m_pop_connection);
}

