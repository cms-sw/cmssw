#ifndef HCALPEDESTALSDBWRITER
#define HCALPEDESTALSDBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalPedestalsSourceHandler.h"


//
// class decleration
//

class HcalPedestalsDBWriter : public popcon::PopConAnalyzer<HcalPedestals>
{
	public:
		HcalPedestalsDBWriter(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
};


#endif
