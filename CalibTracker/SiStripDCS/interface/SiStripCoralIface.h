#ifndef SISTRIPCORALIFACE_H
#define SISTRIPCORALIFACE_H
#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Exception.h"
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
  void doQuery(std::string queryType, const coral::TimeStamp& startTime, const coral::TimeStamp& endTime, std::vector<coral::TimeStamp>&, std::vector<float>&, std::vector<std::string>& );
  /** Method to access the settings for each channel stored in the status change table*/
  void doSettingsQuery(const coral::TimeStamp& startTime,const coral::TimeStamp& endTime,std::vector<coral::TimeStamp>&,std::vector<float>&,std::vector<std::string>&,std::vector<uint32_t>&);
  //
  void doNameQuery(std::vector<std::string> &vec_dpname, std::vector<uint32_t> &vec_dpid);
 private:
  /** Set up the connection to the database*/
  void initialize();

  /* member variables*/
  std::string m_connectionString;
  std::string m_authPath;
  std::map<std::string,unsigned int> m_id_map;
  cond::persistency::Session m_session;
  std::unique_ptr<cond::persistency::TransactionScope> m_transaction;
  // cond::CoralTransaction* m_coraldb;
  // cond::Connection* con;

  bool debug_;
};
#endif
