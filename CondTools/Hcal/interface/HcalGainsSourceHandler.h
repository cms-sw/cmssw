#ifndef HCALGAINSSOURCEHANDLER
#define HCALGAINSSOURCEHANDLER

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/PopCon/interface/LogReader.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
// user include files
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"


namespace popcon
{
	class HcalGainsSourceHandler : public popcon::PopConSourceHandler<HcalGains>
	{

		public:
			void getNewObjects();
			~HcalGainsSourceHandler(); 
			HcalGainsSourceHandler(std::string,std::string,std::string, const edm::Event& evt, const edm::EventSetup& est, std::string); 

		private:
			std::string m_pop_connect; //connect string to popcon metaschema
			std::string m_name;
			std::string m_cs;
			LogReader* lgrdr;
	};
}
#endif
