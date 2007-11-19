#include "CondCore/PopCon/interface/OfflineDBInterface.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"

popcon::OfflineDBInterface::OfflineDBInterface (std::string cstring, std::string catalog) : m_connect(cstring), m_catalog(catalog) {

	session=new cond::DBSession;
	session->configuration().setAuthenticationMethod( cond::XML );
	//session->sessionConfiguration().setMessageLevel( cond::Debug );
	session->configuration().setMessageLevel( cond::Error );
	session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut(10);
	session->configuration().connectionConfiguration()->enableConnectionSharing();
	session->configuration().connectionConfiguration()->enableReadOnlySessionOnUpdateConnections();
}

popcon::OfflineDBInterface::~OfflineDBInterface ()
{
	delete session;
}


//Gets the IOV and payload information from the conditions schema
//returns the list as map<string tag, struct payloaddata>
std::map<std::string, popcon::PayloadIOV> popcon::OfflineDBInterface::getStatusMap()
{

	//TODO - currently all the tags per schema are being returned 
	//Possiblity to return the tags for a given object type??
	getAllTagsInfo();
	return m_status_map;
}

popcon::PayloadIOV popcon::OfflineDBInterface::getSpecificTagInfo(std::string tag)
{
	PayloadIOV dummy = {0,0,0xffffffff,"noTag"};
	getAllTagsInfo();
	if(m_status_map.find(tag) == m_status_map.end())	
		return dummy;
	return m_status_map[tag];
}


void popcon::OfflineDBInterface::getSpecificPayloadMap(std::string){
	//FIXME Implement 
}

//Fetches the list of tags in a schema and returns payload ingormation associated with a tag
void  popcon::OfflineDBInterface::getAllTagsInfo()
{		
  //m_status_map.clear();
  static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
  conHandler.registerConnection(m_connect,m_connect,0);
  if ( !m_status_map.empty() )  return;
  popcon::PayloadIOV piov;
  try{
    
    session->open();
    conHandler.connect(session);
    cond::Connection* myconnection=conHandler.getConnection(m_connect);
    cond::CoralTransaction& coraldb=myconnection->coralTransaction(true);
    cond::MetaData metadata_svc(coraldb);
    std::vector<std::string> alltags;
    std::vector<std::string> alltokens;
    coraldb.start();
    metadata_svc.listAllTags(alltags);
    //std::copy (alltags.begin(),
    //		alltags.end(),
    //		std::ostream_iterator<std::string>(std::cerr,"\n")
    //	  );
    
    //get the pool tokens
    for(std::vector<std::string>::iterator it = alltags.begin(); it != alltags.end(); it++)
      alltokens.push_back( metadata_svc.getToken(*it));
    coraldb.commit();	
    //std::copy (alltokens.begin(),
    //		alltokens.end(),
    //		std::ostream_iterator<std::string>(std::cerr,"\n")
    //	  );
    //connect to pool DB
    cond::PoolTransaction& pooldb=myconnection->poolTransaction(true);
    cond::IOVService iovservice(pooldb);
    pooldb.start();
    cond::IOVIterator* ioviterator;
    std::string payloadContainer;
    unsigned int counter=0;
    unsigned int itpos =0;
    for(std::vector<std::string>::iterator tok_it = alltokens.begin(); tok_it != alltokens.end(); tok_it++, itpos++){
      ioviterator=iovservice.newIOVIterator(*tok_it);
      pooldb.start();
      counter = 0;
      payloadContainer=iovservice.payloadContainerName(*tok_it);
      //std::cerr<<"Tag "<< alltags[itpos]  <<"\n";
      //std::cerr<<"PayloadContainerName "<<payloadContainer<<"\n";
      //std::cerr<<"since \t till \t payloadToken"<<std::endl;
      while( ioviterator->next() ){
	//	std::cout<<ioviterator->validity().first<<" \t "<<ioviterator->validity().second<<" \t "<<ioviterator->payloadToken()<<std::endl;	
	++counter;
      }
      piov.number_of_objects = counter;		
      piov.last_since = ioviterator->validity().first;
      piov.last_till = ioviterator->validity().second;
      piov.container_name  = payloadContainer;
      m_status_map[alltags[itpos]] = piov;
      delete ioviterator;
    }
    pooldb.commit();
  }catch(cond::Exception& er){
    std::cerr<< "Problem accessing the DB, Offline information not available" <<std::endl;
  }catch(std::exception& er){
    std::cerr<< "Offline information not available" <<std::endl;
  }catch(...){
    std::cout<<"Unknown error"<<std::endl;
  } 
}

