#ifndef ECAL_INTERCALIB_HANDLER_H
#define ECAL_INTERCALIB_HANDLER_H

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



#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/EcalCondDB/interface/RunDCSMagnetDat.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"



#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon
{


	class EcalIntercalibHandler : public popcon::PopConSourceHandler<EcalIntercalibConstants>
	{

		public:
                        EcalIntercalibHandler(edm::ParameterSet const & );
			~EcalIntercalibHandler(); 
			
			void getNewObjects();

			std::string id() const { return m_name;}
			EcalCondDBInterface* econn;



		private:
			const EcalIntercalibConstants * myintercalib;

			unsigned int m_firstRun ;
			unsigned int m_lastRun ;
			
			std::string m_location;
			std::string m_gentag;
			std::string m_sid;
			std::string m_user;
			std::string m_pass;
                        std::string m_locationsource;
                        std::string m_name;
                        std::string m_file_lowfield;
                        std::string m_file_highfield;


	};
}
#endif

