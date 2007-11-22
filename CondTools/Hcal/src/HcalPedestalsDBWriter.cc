#include "CondTools/Hcal/interface/HcalPedestalsDBWriter.h"

using namespace popcon;

HcalPedestalsDBWriter::HcalPedestalsDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<HcalPedestals>(ps,"HcalPedestals")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
} 	

void HcalPedestalsDBWriter::HcalPedestalsDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object = new HcalPedestalsSourceHandler("HcalPedestals", m_offline_connection, m_catalog, evt, est, m_pop_connection);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPedestalsDBWriter);

