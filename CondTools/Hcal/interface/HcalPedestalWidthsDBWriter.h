#ifndef HCALPEDESTALWIDTHSDBWRITER
#define HCALPEDESTALWIDTHSDBWRITER

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalPedestalWidthsSourceHandler.h"


//
// class decleration
//

class HcalPedestalWidthsDBWriter : public popcon::PopConAnalyzer<HcalPedestalWidths>
{
	public:
		HcalPedestalWidthsDBWriter(const edm::ParameterSet&);
	private: 
		std::string m_pop_connection;
		void initSource(const edm::Event& evt, const edm::EventSetup& est);
		unsigned int sinceTime, tillTime;
};


#endif
