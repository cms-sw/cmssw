#ifndef OFFLINE_DB_INTERFACE_H
#define OFFLINE_DB_INTERFACE_H


#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include <iterator>
#include <iostream>
#include <string>
#include <map>

namespace popcon
{
	//mapped type for subdetector information on offline db contents 
	struct PayloadIOV
	{
		unsigned int number_of_objects;
		//last payload object IOV info 
		unsigned int last_since;
		unsigned int last_till;
		std::string container_name;	
	};
		

	class OfflineDBInterface
	{
		public:	
			OfflineDBInterface(std::string,std::string);
			virtual ~OfflineDBInterface();
			virtual std::map<std::string, PayloadIOV> getStatusMap();
			PayloadIOV getSpecificTagInfo(std::string tag);
		private:
			//tag - IOV/Payload information map
			std::map<std::string, PayloadIOV> m_status_map;
			std::string m_connect;
			std::string m_catalog;
			std::string m_user;
			std::string m_pass;

			std::string m_payloadName;
			
			cond::DBSession* session;

			void getAllTagsInfo();
			void getSpecificPayloadMap(std::string);
			
	};

}
#endif
