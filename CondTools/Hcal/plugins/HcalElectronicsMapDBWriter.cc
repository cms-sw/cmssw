#include "CondTools/Hcal/interface/HcalElectronicsMapDBWriter.h"

using namespace popcon;

HcalElectronicsMapDBWriter::HcalElectronicsMapDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<HcalElectronicsMap>(ps,"HcalElectronicsMap")
{
  //	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	unsigned int defBeginTime = edm::IOVSyncValue::beginOfTime().eventID().run();
	unsigned int defEndTime = edm::IOVSyncValue::endOfTime().eventID().run();
	sinceTime = ps.getUntrackedParameter<unsigned>("startRun",defBeginTime);
	tillTime = ps.getUntrackedParameter<unsigned>("endRun",defEndTime);
} 	

void HcalElectronicsMapDBWriter::HcalElectronicsMapDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
  this->m_handler_object = new HcalElectronicsMapSourceHandler("HcalElectronicsMap", m_offline_connection, evt, est, sinceTime, tillTime);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalElectronicsMapDBWriter);

