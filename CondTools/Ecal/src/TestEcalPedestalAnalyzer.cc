#include "CondTools/Ecal/interface/TestEcalPedestalAnalyzer.h"


TestEcalPedestalAnalyzer::TestEcalPedestalAnalyzer(const edm::ParameterSet& ps) : PopConAnalyzer<EcalPedestals>(ps,"EcalPedestals")
{
  std::cout << "Implemented EcalPedestalAnalyzer Constructor\n";
  m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
  m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
  m_sid= ps.getParameter<std::string>("OnlineDBSID");
  m_user= ps.getParameter<std::string>("OnlineDBUser");
  m_pass= ps.getParameter<std::string>("OnlineDBPassword");
  m_source= ps.getParameter<std::string>("Source");
  m_location=ps.getParameter<std::string>("Location");
  m_gentag=ps.getParameter<std::string>("GenTag");
      
} 	

void TestEcalPedestalAnalyzer::TestEcalPedestalAnalyzer::initSource(const edm::Event& evt, const edm::EventSetup& est)
{
	std::cerr << m_offline_connection << "initsource*******************************\n";
	if(m_source=="H2") { 
	  this->m_handler_object =new EcalH2PedestalsHandler("EcalPedestals",m_offline_connection, m_catalog,evt,est, m_firstRun,m_lastRun,  m_sid,  m_user,  m_pass );
	} else if (m_source=="H4") { 
	  this->m_handler_object =new EcalH2PedestalsHandler("EcalPedestals",m_offline_connection, m_catalog,evt,est, m_firstRun,m_lastRun,  m_sid,  m_user,  m_pass );
	} else if (m_source=="P5") { 
	  this->m_handler_object =new EcalPedestalsHandler("EcalPedestals",m_offline_connection, m_catalog,evt,est, m_firstRun,m_lastRun,  m_sid,  m_user,  m_pass, m_gentag, m_location );
	} 
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestEcalPedestalAnalyzer);

