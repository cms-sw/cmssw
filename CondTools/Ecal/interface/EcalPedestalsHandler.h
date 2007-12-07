#ifndef ECAL_PEDESTALS_HANDLER_H
#define ECAL_PEDESTALS_HANDLER_H


#include <typeinfo>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"


#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"



#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <time.h>

using namespace std;
using namespace oracle::occi;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon
{
	class EcalPedestalsHandler : public popcon::PopConSourceHandler<EcalPedestals>
	{

		public:
			void getNewObjects();
			~EcalPedestalsHandler(); 
			EcalPedestalsHandler(std::string,std::string,std::string, const edm::Event& evt, const edm::EventSetup& est,
					  unsigned int firstRun,unsigned int lastRun, std::string sid, std::string user, std::string pass, std::string tag, std::string loca); 

			EcalCondDBInterface* econn;
		private:
			const EcalPedestals * mypedestals;

			unsigned long m_firstRun ;
			unsigned long m_lastRun ;
			
			std::string m_location;
			std::string m_gentag;
			std::string m_sid;
			std::string m_user;
			std::string m_pass;

	};
}
#endif
