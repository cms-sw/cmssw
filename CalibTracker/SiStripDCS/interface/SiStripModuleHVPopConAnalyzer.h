#ifndef SISTRIPMODULEHVPOPCON_ANALYZER_H
#define SISTRIPMODULEHVPOPCON_ANALYZER_H

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVHandler.h"
class SiStripModuleHVPopConAnalyzer : public popcon::PopConAnalyzer<SiStripModuleHV>
{
       public:
	     SiStripModuleHVPopConAnalyzer(const edm::ParameterSet&);
       private:
         std::string m_pop_connection;
       	 void initSource(const edm::Event& evt, const edm::EventSetup& est);
//{	
//	    m_handler_object=new SiStripModuleHVHandler("SiStripModuleHV",m_offline_connection, m_catalog,evt,est,m_pop_connection);
//         }
};

#endif
