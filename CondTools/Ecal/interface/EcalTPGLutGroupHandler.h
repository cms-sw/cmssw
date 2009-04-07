#ifndef ECAL_TPG_LUTGROUP_HANDLER_H
#define ECAL_TPG_LUTGROUP_HANDLER_H

#include <vector>
#include <typeinfo>
#include <string>
#include <map>
#include <iostream>
#include <time.h>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"


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

#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"


#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"


using namespace std;
using namespace oracle::occi;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

//class EcalElectronicsMapping;

namespace popcon
{


	class EcalTPGLutGroupHandler : public popcon::PopConSourceHandler<EcalTPGLutGroup>
	{

		public:
                        EcalTPGLutGroupHandler(edm::ParameterSet const & );
			~EcalTPGLutGroupHandler(); 
			
			std::map<string, int> makeTTEBDetId();
			
			void getNewObjects();
			std::string id() const { return m_name;}
			EcalCondDBInterface* econn;

		private:
			const EcalTPGLutGroup * mypedestals;
			unsigned long m_firstRun ;
			unsigned long m_lastRun ;
			std::map <std::string, int> mapDetEBId;
			std::map <std::string, int> mapDetEEId;
			
			const EcalElectronicsMapping* ecalMapping_;
			
			std::string m_location;
			std::string m_gentag;
			std::string m_sid;
			std::string m_user;
			std::string m_pass;
                        std::string m_locationsource;
                        std::string m_name;
			unsigned int m_runnr;



	};
}
#endif

