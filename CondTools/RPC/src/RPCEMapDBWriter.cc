#include "CondTools/RPC/interface/RPCEMapDBWriter.h"

using namespace popcon;

RPCEMapDBWriter::RPCEMapDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<RPCEMap>(ps,"RPCEMap")
{
  m_validate= ps.getUntrackedParameter<int>("Validate",0);
  m_host= ps.getUntrackedParameter<std::string>("OnlineDBHost","lxplus.cern.ch");
  m_sid= ps.getUntrackedParameter<std::string>("OnlineDBSID","blah");
  m_user= ps.getUntrackedParameter<std::string>("OnlineDBUser","blaah");
  m_pass= ps.getUntrackedParameter<std::string>("OnlineDBPass","blaaah");
  m_port= ps.getUntrackedParameter<int>("OnlineDBPort",1521);
} 	

void RPCEMapDBWriter::RPCEMapDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
  this->m_handler_object = new RPCEMapSourceHandler("RPCEMap", m_offline_connection, evt, est, m_validate, m_host, m_sid, m_user, m_pass, m_port);
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCEMapDBWriter);

