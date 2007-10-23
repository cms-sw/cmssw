#ifndef TEST_ECALPED_ANALYZER_H
#define TEST_ECALPED_ANALYZER_H


#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalH2PedestalsHandler.h"
#include "CondTools/Ecal/interface/EcalPedestalsHandler.h"


using namespace popcon;

class TestEcalPedestalAnalyzer : public popcon::PopConAnalyzer<EcalPedestals>
{
	public:
		TestEcalPedestalAnalyzer(const edm::ParameterSet&);
	private: 
		unsigned long m_firstRun ;
		unsigned long m_lastRun ;			
		std::string m_sid;
		std::string m_user;
		std::string m_pass;
		std::string m_source;
		std::string m_gentag;
		std::string m_location;

		void initSource(const edm::Event& evt, const edm::EventSetup& est);


	      
};


#endif
