#ifndef CONDCORE_PLUGINSYSTEM_DATAPROXY_H
#define CONDCORE_PLUGINSYSTEM_DATAPROXY_H
// -*- C++ -*-
//
// Package:     PluginSystem
// Class  :     CondDataProxy
// 
/**\class CondDataProxy CondDataProxy.h CondCore/PluginSystem/interface/CondDataProxy.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:40:27 EDT 2005
// $Id: DataProxy.h,v 1.3 2005/09/01 09:28:59 xiezhen Exp $
//

// system include files
//#include <memory>
#include <iostream>
// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "DataSvc/Ref.h"
#include "DataSvc/RefException.h"

// forward declarations

namespace pool{
  class IDataSvc;
}

namespace cond{
  template< class RecordT, class DataT >
  class DataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT>{
  public:
    DataProxy( pool::IDataSvc* svc, std::map<std::string,std::string>::iterator& pProxyToToken ): m_svc(svc), m_pProxyToToken(pProxyToToken) { 
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
      m_data=*(new pool::Ref<DataT>(m_svc,m_pProxyToToken->second));
      //m_data=pool::Ref<DataT>(m_svc,m_pProxyToToken->second);
      try{
	*m_data;
      }catch( const pool::RefException& er){
	std::cerr<<"caught RefException "<<er.what()<<std::endl;
	throw cms::Exception( er.what() );
      }catch( const seal::Exception& er ){
	std::cerr<<"caught seal Exception "<<er.what()<<std::endl;
	throw cms::Exception( er.what() );
      }catch( ... ){
	throw cms::Exception( "Funny error" );
      }
      return &(*m_data);
    }
    
    virtual void invalidateCache() {
      std::cout<<"invalidateCache"<<std::endl;
      //m_data.clear();
      //std::cout<<"end invalidateCache"<<std::endl;
    }
  private:
    //DataProxy(); // stop default
    const DataProxy& operator=( const DataProxy& ); // stop default
    // ---------- member data --------------------------------
    pool::IDataSvc* m_svc;
    std::map<std::string,std::string>::iterator m_pProxyToToken;
    pool::Ref<DataT> m_data;
  };
}

#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
