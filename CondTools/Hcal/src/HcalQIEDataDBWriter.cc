#include "CondTools/Hcal/interface/HcalQIEDataDBWriter.h"

using namespace popcon;

HcalQIEDataDBWriter::HcalQIEDataDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<HcalQIEData>(ps,"HcalQIEData")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
} 	

void HcalQIEDataDBWriter::HcalQIEDataDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object = new HcalQIEDataSourceHandler("HcalQIEData", m_offline_connection, m_catalog, evt, est, m_pop_connection);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalQIEDataDBWriter);

