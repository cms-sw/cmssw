#ifndef HCALQIEDATADBWRITER
#define HCALQIEDATADBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalQIEDataSourceHandler.h"


//
// class decleration
//

class HcalQIEDataDBWriter : public popcon::PopConAnalyzer<HcalQIEData>
{
	public:
		HcalQIEDataDBWriter(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
