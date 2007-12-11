#ifndef HCALQIEDATASOURCEHANDLER
#define HCALQIEDATASOURCEHANDLER

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/PopCon/interface/LogReader.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
// user include files
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"


namespace popcon
{
	class HcalQIEDataSourceHandler : public popcon::PopConSourceHandler<HcalQIEData>
	{

		public:
			void getNewObjects();
			~HcalQIEDataSourceHandler(); 
			HcalQIEDataSourceHandler(const std::string&, const std::string&, const edm::Event& evt, const edm::EventSetup& est, unsigned int, unsigned int); 

		private:
			std::string m_pop_connect; //connect string to popcon metaschema
			std::string m_name;
			std::string m_cs;
			LogReader* lgrdr;
			unsigned int snc, tll;
	};
}
#endif
