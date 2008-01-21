#include "CondTools/RPC/interface/RPCReadOutMappingDBWriter.h"

using namespace popcon;

RPCReadOutMappingDBWriter::RPCReadOutMappingDBWriter(const edm::ParameterSet& ps) : PopConAnalyzer<RPCReadOutMapping>(ps,"RPCReadOutMapping")
{
  m_validate= ps.getUntrackedParameter<int>("Validate",0);
  m_host= ps.getUntrackedParameter<std::string>("OnlineDBHost","lxplus.cern.ch");
  m_sid= ps.getUntrackedParameter<std::string>("OnlineDBSID","blah");
  m_user= ps.getUntrackedParameter<std::string>("OnlineDBUser","blaah");
  m_pass= ps.getUntrackedParameter<std::string>("OnlineDBPass","blaaah");
  m_port= ps.getUntrackedParameter<int>("OnlineDBPort",1521);
} 	

void RPCReadOutMappingDBWriter::RPCReadOutMappingDBWriter::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
  this->m_handler_object = new RPCReadOutMappingSourceHandler("RPCReadOutMapping", m_offline_connection, evt, est, m_validate, m_host, m_sid, m_user, m_pass, m_port);
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCReadOutMappingDBWriter);

