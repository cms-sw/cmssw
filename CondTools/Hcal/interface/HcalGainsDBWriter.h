#ifndef HCALGAINSDBWRITER
#define HCALGAINSDBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalGainsSourceHandler.h"


//
// class decleration
//

class HcalGainsDBWriter : public popcon::PopConAnalyzer<HcalGains>
{
	public:
		HcalGainsDBWriter(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
