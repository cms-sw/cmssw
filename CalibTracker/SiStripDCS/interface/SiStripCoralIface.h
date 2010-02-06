#ifndef SISTRIPCORALIFACE_H
#define SISTRIPCORALIFACE_H
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CoralBase/TimeStamp.h"

#include <iterator>
#include <iostream>
#include <string>
#include <map>


/**	
   \class SiStripCoralIface
   \brief An interface class to the PVSS cond DB
   \author J.Chen, J.Cole
*/
class SiStripCoralIface
{
 public:	
  /** constructor */
  SiStripCoralIface(std::string connectionString, std::string authenticationPath);
  /** destructor*/
  ~SiStripCoralIface();
  /** Method to retrieve information from status change table or lastValue table.  queryType defines which table is to be accessed.*/
  void doQuery(std::string queryType, coral::TimeStamp startTime, coral::TimeStamp endTime, std::vector<coral::TimeStamp>&, std::vector<float>&, std::vector<std::string>& );
  /** Method to access the settings for each channel stored in the status change table*/
  void doSettingsQuery(coral::TimeStamp startTime,coral::TimeStamp endTime,std::vector<coral::TimeStamp>&,std::vector<float>&,std::vector<std::string>&,std::vector<uint32_t>&);
 private:
  /** Set up the connection to the database*/
  void initialize();

  /* member variables*/
  std::string m_connect;
  std::map<std::string,unsigned int> m_id_map;
  cond::DBSession* session;
  cond::CoralTransaction* m_coraldb;
  cond::Connection* con;
};
#endif
