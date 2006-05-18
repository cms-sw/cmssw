#ifndef CONDCORE_PLUGINSYSTEM_DATAPROXY_H
#define CONDCORE_PLUGINSYSTEM_DATAPROXY_H
//#include <memory>
#include <iostream>
// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "DataSvc/Ref.h"
#include "DataSvc/RefException.h"
#include "DataSvc/IDataSvc.h"

namespace cond{
  template< class RecordT, class DataT >
  class DataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT>{
  public:
    /*  DataProxy( pool::IDataSvc* svc, std::map<std::string,std::string>::iterator& pProxyToToken ): m_svc(svc), m_pProxyToToken(pProxyToToken) { 
      //NOTE: We do this so that the type 'DataT' will get registered
      // when the plugin is dynamically loaded
      edm::eventsetup::DataKey::makeTypeTag<DataT>(); 
    }
    */
    DataProxy( cond::DBSession* session, std::map<std::string,std::string>::iterator& pProxyToToken ): m_session(session), m_pProxyToToken(pProxyToToken) { 
      //NOTE: We do this so that the type 'DataT' will get registered
      // when the plugin is dynamically loaded
      edm::eventsetup::DataKey::makeTypeTag<DataT>(); 
    }
    //virtual ~DataProxy();
    
    // ---------- const member functions ---------------------
    
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    
  protected:
    virtual const DataT* make(const RecordT&, const edm::eventsetup::DataKey&) {
      try{
	m_session->startReadOnlyTransaction();
	m_data=pool::Ref<DataT>(&(m_session->DataSvc()),m_pProxyToToken->second);
	*m_data;
	m_session->commit();
      }catch( const cond::Exception& er ){
	throw er;
      }catch( const pool::RefException& er ){
	//std::cerr<<"caught RefException "<<er.what()<<std::endl;
	throw cond::Exception( er.what() );
      }catch( const pool::Exception& er ){
	//std::cerr<<"caught pool Exception "<<er.what()<<std::endl;
	throw cond::Exception( er.what() );
      }catch( const std::exception& er ){
        //std::cerr<<"caught std Exception "<<er.what()<<std::endl;
        throw cond::Exception( er.what() );
      }catch( ... ){
	throw cond::Exception( "Funny error" );
      }
      return &(*m_data);
    }
    virtual void invalidateCache() {
      m_data.clear();
      //std::cout<<"end invalidateCache"<<std::endl;
    }
  private:
    //DataProxy(); // stop default
    const DataProxy& operator=( const DataProxy& ); // stop default
    // ---------- member data --------------------------------
    //pool::IDataSvc* m_svc;
    cond::DBSession* m_session;
    std::map<std::string,std::string>::iterator m_pProxyToToken;
    pool::Ref<DataT> m_data;
  };
}

#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
