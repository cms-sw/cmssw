#include "CondTools/Hcal/interface/HcalPedestalsDBWriter.h"

using namespace popcon;

HcalPedestalsDBWriter::HcalPedestalsDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<HcalPedestals>(ps,"HcalPedestals")
{
	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	unsigned int defBeginTime = edm::IOVSyncValue::beginOfTime().eventID().run();
	unsigned int defEndTime = edm::IOVSyncValue::endOfTime().eventID().run();
	sinceTime = ps.getUntrackedParameter<unsigned>("startRun",defBeginTime);
	tillTime = ps.getUntrackedParameter<unsigned>("endRun",defEndTime);
} 	

void HcalPedestalsDBWriter::HcalPedestalsDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object = new HcalPedestalsSourceHandler("HcalPedestals", m_offline_connection, m_catalog, evt, est, m_pop_connection, sinceTime, tillTime);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPedestalsDBWriter);

