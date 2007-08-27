#include "CondCore/PopCon/interface/OfflineDBInterface.h"


popcon::OfflineDBInterface::OfflineDBInterface (std::string cstring, std::string catalog) : m_connect(cstring), m_catalog(catalog) {

	//TODO db passwd, catalog etc should be parametrized

	session=new cond::DBSession(true);
	session->sessionConfiguration().setAuthenticationMethod( cond::XML );
	//session->sessionConfiguration().setMessageLevel( cond::Debug );
	session->sessionConfiguration().setMessageLevel( cond::Error );
	session->connectionConfiguration().setConnectionRetrialTimeOut( 600 );
	session->connectionConfiguration().enableConnectionSharing();
	session->connectionConfiguration().enableReadOnlySessionOnUpdateConnections();
	//std::string userenv(std::string("CORAL_AUTH_USER=")+m_user);
	//std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+m_pass);
	//::putenv(const_cast<char*>(userenv.c_str()));
	//::putenv(const_cast<char*>(passenv.c_str()));
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
	if (!m_status_map.empty())
		return;
	popcon::PayloadIOV piov;
	try{

		session->open();
		cond::RelationalStorageManager coraldb(m_connect,session);
		cond::MetaData metadata_svc(coraldb);
		std::vector<std::string> alltags;
		std::vector<std::string> alltokens;

		coraldb.connect(cond::ReadOnly);
		coraldb.startTransaction(true);
		metadata_svc.listAllTags(alltags);
		//std::copy (alltags.begin(),
		//		alltags.end(),
		//		std::ostream_iterator<std::string>(std::cerr,"\n")
		//	  );

		//get the pool tokens
		for(std::vector<std::string>::iterator it = alltags.begin(); it != alltags.end(); it++)
			alltokens.push_back( metadata_svc.getToken(*it));
		coraldb.commit();	
		coraldb.disconnect();	

		//std::copy (alltokens.begin(),
		//		alltokens.end(),
		//		std::ostream_iterator<std::string>(std::cerr,"\n")
		//	  );


		//connect to pool DB
		cond::PoolStorageManager pooldb(m_connect,m_catalog,session);
		cond::IOVService iovservice(pooldb);
		cond::IOVIterator* ioviterator;
		std::string payloadContainer;
		unsigned int counter=0;
		unsigned int itpos =0;
		pooldb.connect();
		for(std::vector<std::string>::iterator tok_it = alltokens.begin(); tok_it != alltokens.end(); tok_it++, itpos++){
			ioviterator=iovservice.newIOVIterator(*tok_it);
			pooldb.startTransaction(true);
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
		pooldb.disconnect();
		session->close();


	}catch(cond::Exception& er){
		std::cerr<< "Problem accessing the DB, Offline information not available" <<std::endl;
	}catch(std::exception& er){
		std::cerr<< "Offline information not available" <<std::endl;
	}catch(...){
		std::cout<<"Unknown error"<<std::endl;
	}

}

