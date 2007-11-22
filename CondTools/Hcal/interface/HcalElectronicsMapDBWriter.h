#ifndef HCALELECTRONICSMAPDBWRITER
#define HCALELECTRONICSMAPDBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalElectronicsMapSourceHandler.h"


//
// class decleration
//

class HcalElectronicsMapDBWriter : public popcon::PopConAnalyzer<HcalElectronicsMap>
{
	public:
		HcalElectronicsMapDBWriter(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
