#ifndef POPCON_LOGGER_H
#define POPCON_LOGGER_H

#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include <iterator>
#include <iostream>
#include <string>
#include <map>

namespace popcon
{

	class Logger
	{
		public:	
			Logger(std::string connectionString, std::string offlineString, std::string payloadName, bool debug);
			virtual ~Logger();

			void finalizePayload(std::string ok="OK");
			void finalizeExecution(std::string ok="OK");
			void newPayload();
			void newExecution();
			void lock();
			void unlock();

		private:
			void initialize();
			void updateExecID();
			void updatePayloadID();
			void payloadIDMap();
			void disconnect();

			std::string m_obj_name;
			std::string m_connect;
			std::string m_offline;
			
			bool m_debug;
			bool m_established;
			bool m_sqlite;

			unsigned int m_exec_id;
			unsigned int m_payload_id;

			std::map<std::string,unsigned int> m_id_map;

			cond::DBSession* session;
			cond::RelationalStorageManager* m_coraldb;
	};

}
#endif
