//#include <iostream>
#include "CondCore/PopCon/interface/OfflineDBInterface.h"
//#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
//static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
popcon::OfflineDBInterface::OfflineDBInterface (const std::string& connect ) : m_connect(connect) {
}

popcon::OfflineDBInterface::~OfflineDBInterface ()
{
}


//Gets the IOV and payload information from the conditions schema
//returns the list as map<string tag, struct payloaddata>
OfflineDBInterface::States const & popcon::OfflineDBInterface::getStatusMap() const
{

  //TODO - currently all the tags per schema are being returned 
  //Possiblity to return the tags for a given object type??
  getAllTagsInfo();
  return m_status_map;
}

popcon::PayloadIOV popcon::OfflineDBInterface::getSpecificTagInfo(const std::string& tag) const 
{
  static const PayloadIOV dummy = {0,0,0xffffffff,"noTag"};
  getAllTagsInfo();
  States::const_iterator p = m_status_map.find(tag);
  return (p == m_status_map.end()) ?
    dummy :
    (*p).second;
}

void popcon::OfflineDBInterface::getSpecificPayloadMap(const std::string&) const {
	//FIXME Implement 
}

//Fetches the list of tags in a schema and returns payload ingormation associated with a tag
void  popcon::OfflineDBInterface::getAllTagsInfo() const
{		
  cond::DBSession session;  
  session.configuration().setAuthenticationMethod( cond::XML );
  session.configuration().setMessageLevel( cond::Error );

  //m_status_map.clear();
  if ( !m_status_map.empty() )  return;
  popcon::PayloadIOV piov;
  session.open();
  cond::Connection con(m_connect,-1);
  con.connect(&session);
  cond::CoralTransaction& coraldb=con.coralTransaction();
  cond::MetaData metadata_svc(coraldb);
  std::vector<std::string> alltags;
  std::vector<std::string> alltokens;
  coraldb.start(false);
  metadata_svc.listAllTags(alltags);
  //get the pool tokens
  for(std::vector<std::string>::const_iterator it = alltags.begin(); it != alltags.end(); it++){
    alltokens.push_back( metadata_svc.getToken(*it));
  }
  coraldb.commit();	
  //std::copy (alltokens.begin(),
  //		alltokens.end(),
  //		std::ostream_iterator<std::string>(std::cerr,"\n")
  //	  );
  //connect to pool DB
  cond::PoolTransaction& pooldb=con.poolTransaction();
  cond::IOVService iovservice(pooldb);
  pooldb.start(true);
  size_t itpos=0;
  for(std::vector<std::string>::iterator tok_it = alltokens.begin(); tok_it != alltokens.end(); tok_it++, itpos++){
    std::auto_ptr<cond::IOVIterator> ioviterator(iovservice.newIOVIterator(*tok_it,cond::IOVService::backwardIter));
    ioviterator->next();
    piov.number_of_objects = ioviterator->size();		
    piov.last_since = ioviterator->validity().first;
    piov.last_till = ioviterator->validity().second;
    piov.container_name  = iovservice.payloadContainerName(*tok_it);
    m_status_map[alltags[itpos]] = piov;
  }
  pooldb.commit();
  con.disconnect();
}

