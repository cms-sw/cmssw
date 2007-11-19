#ifndef CONDCORE_POPCON_LOGREADER_H
#define CONDCORE_POPCON_LOGREADER_H

#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "CoralBase/Exception.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/TimeStamp.h"

#include <iterator>
#include <iostream>
#include <string>
#include <map>
namespace cond{
  class CoralTransaction;
}
namespace popcon
{


	class LogReader
	{
		public:	
			LogReader(std::string connectionString);
			virtual ~LogReader();
			coral::TimeStamp lastRun(std::string&,std::string&);
			
		private:
			void initialize();
			
			std::string m_connect;
			cond::DBSession* session;
			cond::CoralTransaction* m_coraldb;
	};
}
#endif
