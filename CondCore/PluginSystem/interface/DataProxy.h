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
// $Id$
//

// system include files
//#include <memory>
#include <iostream>
// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "DataSvc/Ref.h"
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
      std::cout<<"DataProxy::make"<<std::endl;
      std::cout<<"my token is "<<m_pProxyToToken->second<<std::endl;
      pool::Ref<DataT> mydata(m_svc,m_pProxyToToken->second);
      std::cout<<"hello" <<std::endl;
      return &(*mydata);
    }
    
    virtual void invalidateCache() {
      std::cout<<"DataProxy::invalidateCache, do nothing"<<std::endl;
    }
  private:
    //DataProxy(); // stop default
    const DataProxy& operator=( const DataProxy& ); // stop default
    // ---------- member data --------------------------------
    pool::IDataSvc* m_svc;
    std::map<std::string,std::string>::iterator m_pProxyToToken;
  };
}

#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
