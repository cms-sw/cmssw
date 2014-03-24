#ifndef CondCore_CondDB_PayloadProxy_h
#define CondCore_CondDB_PayloadProxy_h

#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Time.h"

namespace cond {

  namespace persistency {

    // still needed??
    /* get iov by name (key, tag, ...)
     */
    class CondGetter {
    public:
      virtual ~CondGetter(){}
      virtual IOVProxy get(std::string name) const=0;
      
    };
    
    // implements the not templated part...
    class BasePayloadProxy {
    public:
      
      // 
      BasePayloadProxy();

      void setUp( Session dbSession );
      
      void loadTag( const std::string& tag );
      
      void reload();
      
      virtual ~BasePayloadProxy();

      virtual void make()=0;
      
      virtual void invalidateCache()=0;
      
      // current cached object token
      const Hash& payloadId() const { return m_currentIov.payloadId;}
      
      // this one had the loading in a separate function in the previous impl
      ValidityInterval setIntervalFor( Time_t target, bool loadPayload=false );
      
      bool isValid() const;
      
      TimeType timeType() const { return m_iovProxy.timeType();}
      
      virtual void loadMore(CondGetter const &){
      }

      IOVProxy iov();
      
      const std::vector<Iov_t>& requests() const {
	return m_requests;
      }
    
    private:
      virtual void loadPayload() = 0;   
      
    
    protected:
      IOVProxy m_iovProxy;
      Iov_t m_currentIov;
      Session m_session;
      std::vector<Iov_t> m_requests;
      
    };
    

    /* proxy to the payload valid at a given time...
       
    */
    template<typename DataT>
    class PayloadProxy : public BasePayloadProxy {
    public:
      
      explicit PayloadProxy( const char * source=0 ) :
	BasePayloadProxy() {}
      
      virtual ~PayloadProxy(){}
      
      // dereference 
      const DataT & operator()() const {
	if( !m_data ) {
	  throwException( "The Payload has not been loaded.","PayloadProxy::operator()");
	}
	return (*m_data); 
      }

      virtual void make(){
	if( isValid() ){
	  if( m_currentIov.payloadId == m_currentPayloadId ) return;
	  m_session.transaction().start(true);
	  loadPayload();
	  m_session.transaction().commit();
	}
      }
      
      virtual void invalidateCache() {
	m_data.reset();
	m_currentPayloadId.clear();
	m_currentIov.clear();
	m_requests.clear();
      }

    protected:
      virtual void loadPayload() {
	if( m_currentIov.payloadId.empty() ){
	  throwException( "Can't load payload: no valid IOV found.","PayloadProxy::loadPayload" );
	}
	m_data = m_session.fetchPayload<DataT>( m_currentIov.payloadId );
	m_currentPayloadId = m_currentIov.payloadId;	  
	m_requests.push_back( m_currentIov );
      }
      
    private:
      boost::shared_ptr<DataT> m_data;
      Hash m_currentPayloadId;
    };
    
  }
}
#endif // CondCore_CondDB_PayloadProxy_h
