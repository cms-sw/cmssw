#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVPopConAnalyzer.h"

using namespace popcon;

SiStripModuleHVPopConAnalyzer::SiStripModuleHVPopConAnalyzer(const edm::ParameterSet& ps): PopConAnalyzer<SiStripModuleHV>(ps,"SiStripModuleHV")
{

m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");

}
void 
SiStripModuleHVPopConAnalyzer::SiStripModuleHVPopConAnalyzer::initSource(const 
edm::Event& evt, const edm::EventSetup& est)
{
 this->m_handler_object =new SiStripModuleHVHandler("SiStripModuleHV",m_offline_connection, m_catalog,evt,est, m_pop_connection);
}

DEFINE_FWK_MODULE(SiStripModuleHVPopConAnalyzer);
