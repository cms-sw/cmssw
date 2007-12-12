#include "CondTools/Hcal/interface/HcalPedestalWidthsDBWriter.h"

using namespace popcon;

HcalPedestalWidthsDBWriter::HcalPedestalWidthsDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<HcalPedestalWidths>(ps,"HcalPedestalWidths")
{
  //	m_pop_connection = ps.getParameter<std::string> ("popConDBSchema");
	unsigned int defBeginTime = edm::IOVSyncValue::beginOfTime().eventID().run();
	unsigned int defEndTime = edm::IOVSyncValue::endOfTime().eventID().run();
	sinceTime = ps.getUntrackedParameter<unsigned>("startRun",defBeginTime);
	tillTime = ps.getUntrackedParameter<unsigned>("endRun",defEndTime);
} 	

void HcalPedestalWidthsDBWriter::HcalPedestalWidthsDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	this->m_handler_object = new HcalPedestalWidthsSourceHandler("HcalPedestalWidths", m_offline_connection, evt, est, sinceTime, tillTime);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPedestalWidthsDBWriter);

