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
      virtual std::string getTag(std::string name) const=0;
      
    };
    
    // implements the not templated part...
    class BasePayloadProxy {
    public:
      
      // 
      explicit BasePayloadProxy( Session& session );
      
      // 
      BasePayloadProxy( Session& session, const std::string& tag );
      
      void loadTag( const std::string& tag );
      
      void reload();
      
      virtual ~BasePayloadProxy();
      
      virtual void invalidateCache()=0;
      
      // current cached object token
      const Hash& payloadId() const { return m_currentIov.payloadId;}
      
      // this one had the loading in a separate function in the previous impl
      ValidityInterval setIntervalFor( Time_t target );
      
      bool isValid() const;
      
      TimeType timeType() const { return m_iovProxy.timeType();}
      
      virtual void loadMore(CondGetter const &){}
      
    private:
      virtual void loadPayload() = 0;   
      
    
    protected:
      IOVProxy m_iovProxy;
      Iov_t m_currentIov;
      Session m_session;
      
    };
    

    /* proxy to the payload valid at a given time...
       
    */
    template<typename DataT>
    class PayloadProxy : public BasePayloadProxy {
    public:
      
      PayloadProxy( Session& session, const char * source=0 ) :
	BasePayloadProxy( session ) {}
      
      virtual ~PayloadProxy(){}
      
      // dereference 
      const DataT & operator()() const {
	if( !m_data ) {
	  throwException( "The Payload has not been loaded.","PayloadProxy::operator()");
	}
	return (*m_data); 
      }
      
      virtual void invalidateCache() {
	m_data.reset();
	m_currentIov.clear();
      }
      
    
    protected:
      virtual void loadPayload() {
	m_data = m_session.fetchPayload<DataT>( m_currentIov.payloadId );
      }
      
    private:
      boost::shared_ptr<DataT> m_data;
    };
    
  }
}
#endif // CondCore_CondDB_PayloadProxy_h
