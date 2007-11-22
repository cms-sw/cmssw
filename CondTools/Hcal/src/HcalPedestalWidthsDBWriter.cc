#include "CondTools/Hcal/interface/HcalPedestalWidthsDBWriter.h"

using namespace popcon;

HcalPedestalWidthsDBWriter::HcalPedestalWidthsDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<HcalPedestalWidths>(ps,"HcalPedestalWidths")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
} 	

void HcalPedestalWidthsDBWriter::HcalPedestalWidthsDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object = new HcalPedestalWidthsSourceHandler("HcalPedestalWidths", m_offline_connection, m_catalog, evt, est, m_pop_connection);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPedestalWidthsDBWriter);

