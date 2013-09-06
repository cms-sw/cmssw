#ifndef CondCore_CondDB_PayloadProxy_h
#define CondCore_CondDB_PayloadProxy_h
//
// Package:     CondDB
// Class  :     PayloadProxy
// 
/**\class PayloadProxy PayloadProxy.h CondCore/CondDB/interface/PayloadProxy.h
   Description: service for efficient read access to the condition Payloads.  
*/
//
// Author:      Giacomo Govi
// Created:     May 2013
//

//#include "CondCore/CondDB/interface/IOVProxy.h"
//#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/CondDB.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <boost/shared_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

//namespace new_impl {
namespace conddb {

  // value semantics...
  template <typename T> class PayloadProxy {
  public:
    // here the session is a parameter
    explicit PayloadProxy( Session& session );

    //
    PayloadProxy( const PayloadProxy& rhs );

    //
    PayloadProxy& operator=( const PayloadProxy& rhs );

    // set up the iov access to the specified tag
    void load( const std::string& tag );

    // reset the iov access to the current tag
    void reload();

    // clear all the caches
    void reset();

    // search a payload valid for the specified time
    // does not query the DB if a valid object is in the cache already
    boost::shared_ptr<T> get( conddb::Time_t targetTime );

  private:
    
    Session m_session;
    IOVProxy m_iov;
    boost::shared_ptr<conddb::Iov_t> m_current;
    boost::shared_ptr<T> m_cache;
  };

  // implementation

  template <typename T> inline PayloadProxy<T>::PayloadProxy( Session& session ):
    m_session( session ),
    m_iov( session.iovProxy() ),
    m_current( new conddb::Iov_t),
    m_cache(){
    m_current->clear();
  }

  template <typename T> inline PayloadProxy<T>::PayloadProxy( const PayloadProxy& rhs ):
    m_session( rhs.m_session ),
    m_iov( rhs.m_iov ),
    m_current( rhs.m_current ),
    m_cache( rhs.m_cache ){
  }

  template <typename T> inline PayloadProxy<T>& PayloadProxy<T>::operator=( const PayloadProxy& rhs ){
    m_session = rhs.m_session;
    m_iov = rhs.m_iov;
    m_current = rhs.m_current;
    m_cache = rhs.m_cache;
    return *this;
  }

  template <typename T> inline void PayloadProxy<T>::load( const std::string& tag ){
    m_current->clear();
    m_cache.reset();
    m_iov.load( tag );
    std::string payloadType = m_iov.payloadObjectType();
    if( m_iov.payloadObjectType() != conddb::demangledName( typeid(T) ) ) {
      reset();      
      conddb::throwException("Type mismatch: type "+payloadType+
			     "defined for tag "+tag+" is different from the target type.",
			     "PayloadProxy::load");
    }
  }

  template <typename T> inline void PayloadProxy<T>::reload(){
    m_current->clear();
    m_cache.reset();
    m_iov.reload();
  }

  template <typename T> inline void PayloadProxy<T>::reset(){
    m_iov.reset();
    m_current->clear();
    m_cache.reset();
  }

  template <typename T> inline boost::shared_ptr<T> PayloadProxy<T>::get( conddb::Time_t targetTime ){
    //  check if the current iov loaded is the good one...
    if( targetTime < m_current->since || targetTime > m_current->till ){

      // a new payload is required!
      auto iIov = m_iov.find( targetTime );
      if(iIov == m_iov.end() ) conddb::throwException("No iov available for the specified target time.","PayloadProxy::get");
      *m_current = *iIov;

      // finally load the new payload into the cache
      m_cache = m_session.fetchPayload<T>( m_current->payloadId );
    } 
    return m_cache;
  }

}

#endif

