#ifndef CSC_DBGAINS_SRC_IMPL_H
#define CSC_DBGAINS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/PopCon/interface/LogReader.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CSCGainsDBConditions.h"

namespace popcon
{
	class CSCDBGainsImpl : public popcon::PopConSourceHandler<CSCDBGains>
	{

		public:
			void getNewObjects();
			~CSCDBGainsImpl(); 
			CSCDBGainsImpl(const std::string&,
				       const std::string&,
				       const edm::Event& evt, 
				       const edm::EventSetup& est, 
				       const std::string&);
		

		private:
			std::string m_pop_connect; //connect string to popcon metaschema
			std::string m_name;
			std::string m_cs;
			const CSCDBGains * mygains;
			LogReader* lgrdr;
	};
}
#endif
