#ifndef HCALPEDESTALSSOURCEHANDLER
#define HCALPEDESTALSSOURCEHANDLER

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
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"


namespace popcon
{
	class HcalPedestalsSourceHandler : public popcon::PopConSourceHandler<HcalPedestals>
	{

		public:
			void getNewObjects();
			~HcalPedestalsSourceHandler(); 
			HcalPedestalsSourceHandler(std::string,std::string,std::string, const edm::Event& evt, const edm::EventSetup& est, std::string); 

		private:
			std::string m_pop_connect; //connect string to popcon metaschema
			std::string m_name;
			std::string m_cs;
			LogReader* lgrdr;
	};
}
#endif
