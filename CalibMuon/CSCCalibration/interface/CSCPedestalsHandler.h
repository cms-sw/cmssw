#ifndef CSC_DBPEDESTALS_SRC_IMPL_H
#define CSC_DBPEDESTALS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/PopCon/interface/LogReader.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CSCPedestalsDBConditions.h"

namespace popcon
{
	class CSCDBPedestalsImpl : public popcon::PopConSourceHandler<CSCDBPedestals>
	{

		public:
			void getNewObjects();
			~CSCDBPedestalsImpl(); 
			CSCDBPedestalsImpl(std::string,std::string,std::string, const edm::Event& evt, const edm::EventSetup& est, std::string);
		

		private:
			std::string m_pop_connect; //connect string to popcon metaschema
			std::string m_name;
			std::string m_cs;
			//const CSCPedestals * mypedestals;
			const CSCDBPedestals * mypedestals;
			LogReader* lgrdr;
	};
}
#endif
