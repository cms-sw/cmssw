#ifndef SISTRIPCORALIFACE_H
#define SISTRIPCORALIFACE_H
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
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
  SiStripCoralIface( std::string connectionString , std::string authenticationPath, const bool debug);
  /** destructor*/
  ~SiStripCoralIface();
  /** Method to retrieve information from status change table or lastValue table.  queryType defines which table is to be accessed.*/
  void doQuery(std::string queryType, coral::TimeStamp startTime, coral::TimeStamp endTime, std::vector<coral::TimeStamp>&, std::vector<float>&, std::vector<std::string>& );
  /** Method to access the settings for each channel stored in the status change table*/
  void doSettingsQuery(coral::TimeStamp startTime,coral::TimeStamp endTime,std::vector<coral::TimeStamp>&,std::vector<float>&,std::vector<std::string>&,std::vector<uint32_t>&);
  //
  void doNameQuery(std::vector<std::string> &vec_dpname, std::vector<uint32_t> &vec_dpid);
 private:
  /** Set up the connection to the database*/
  void initialize();

  /* member variables*/
  std::string m_connectionString;
  std::map<std::string,unsigned int> m_id_map;
  // cond::DBSession* session;
  cond::DbConnection m_connection;
  cond::DbSession m_session;
  // cond::CoralTransaction* m_coraldb;
  // cond::Connection* con;
  std::auto_ptr<cond::DbScopedTransaction> m_transaction;

  bool debug_;
};
#endif
