#include "CondTools/RPC/interface/L1RPCConfigDBWriter.h"

using namespace popcon;

L1RPCConfigDBWriter::L1RPCConfigDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<L1RPCConfig>(ps,"L1RPCConfig")
{
  m_validate= ps.getUntrackedParameter<int>("Validate",0);
  m_ppt= ps.getUntrackedParameter<int>("PACsPerTower");
  m_dataDir= ps.getUntrackedParameter<std::string>("filedir");
} 	

void L1RPCConfigDBWriter::L1RPCConfigDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
  this->m_handler_object = new L1RPCConfigSourceHandler("L1RPCConfig", m_offline_connection, evt, est, m_validate, m_ppt, m_dataDir);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1RPCConfigDBWriter);

