#ifndef ECAL_DCS_HANDLER_H
#define ECAL_DCS_HANDLER_H

#include <vector>
#include <typeinfo>
#include <string>
#include <map>
#include <iostream>
#include <ctime>

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


#include "OnlineDB/EcalCondDB/interface/RunDCSHVDat.h"
#include "OnlineDB/EcalCondDB/interface/RunDCSLVDat.h"
#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon
{


	class EcalDCSHandler : public popcon::PopConSourceHandler<EcalDCSTowerStatus>
	{

		public:
                        EcalDCSHandler(edm::ParameterSet const & );
			~EcalDCSHandler() override; 
			void printHVDataSet( const std::map<EcalLogicID, RunDCSHVDat>* dataset,int ) const;
			void printLVDataSet( const std::map<EcalLogicID, RunDCSLVDat>* dataset,int ) const ;
			uint16_t  updateHV( RunDCSHVDat* hv, uint16_t dbStatus, int modo=0) const ; 
			uint16_t  updateLV( RunDCSLVDat* lv, uint16_t dbStatus) const ; 
			bool  insertHVDataSetToOffline( const std::map<EcalLogicID, RunDCSHVDat>* dataset, EcalDCSTowerStatus* dcs_temp ) const;
			bool  insertLVDataSetToOffline( const std::map<EcalLogicID, RunDCSLVDat>* dataset, EcalDCSTowerStatus* dcs_temp, const std::vector<EcalLogicID>& ) const;

			void getNewObjects() override;
			std::string id() const override { return m_name;}
			EcalCondDBInterface* econn;

			int * HVLogicIDToDetID(int, int) const;
			int * HVEELogicIDToDetID(int, int) const;
			int * LVLogicIDToDetID(int, int) const;

			int detIDToLogicID(int, int, int);
			uint16_t OffDBStatus( uint16_t dbStatus , int pos ) ;

		private:

			unsigned long m_firstRun ;
			unsigned long m_lastRun ;
			
			std::string m_sid;
			std::string m_user;
			std::string m_pass;
			std::string m_name;
			
	};
}
#endif

