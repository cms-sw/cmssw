#ifndef STATECREATOR_H
#define STATECERATOR_H

#include "CondCore/PopCon/interface/DBState.h"
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
#include <vector>



namespace popcon{

	class StateCreator 
	{

		public:
			StateCreator(std::string connectionString, std::string offlineString, std::string oname, bool debug);
			virtual ~StateCreator();

			bool checkAndCompareState();
			bool previousExceptions(bool& fix);
			void setException(std::string ex);

			//TODO uncomment the following
			//private:
			void getStoredStatusData();
			void getPoolTableName();
			void storeStatusData();
			bool compareStatusData();
			void generateStatusData();
	
			void initialize();
			void disconnect();
		private:

			bool m_sqlite;
			
			//name of the schema (to be determined by object name)
			DBInfo nfo;
			DBState m_saved_state;
			DBState m_current_state;

			std::string m_connect;
			std::string m_offline;
			bool m_debug;
			std::string m_obj_name;
			
			cond::DBSession* session;

			cond::DBSession* condsession;
			
			cond::RelationalStorageManager* m_coraldb;
			coral::ISessionProxy* m_proxy;


	};
}
#endif
