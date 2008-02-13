#ifndef HCALPEDESTALWIDTHSSOURCEHANDLER
#define HCALPEDESTALWIDTHSSOURCEHANDLER

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
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"


namespace popcon
{
	class HcalPedestalWidthsSourceHandler : public popcon::PopConSourceHandler<HcalPedestalWidths>
	{

		public:
			void getNewObjects();
			~HcalPedestalWidthsSourceHandler(); 
			HcalPedestalWidthsSourceHandler(std::string,std::string,std::string, const edm::Event& evt, const edm::EventSetup& est, std::string, unsigned int, unsigned int); 

		private:
			std::string m_pop_connect; //connect string to popcon metaschema
			std::string m_name;
			std::string m_cs;
			LogReader* lgrdr;
			unsigned int snc, tll;
	};
}
#endif
