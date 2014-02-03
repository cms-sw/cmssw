#ifndef CondCore_CondDB_PayloadProxy_h
#define CondCore_CondDB_PayloadProxy_h

#include "CondCore/CondDB/interface/ORAWrapper.h"
#include "CondCore/CondDB/interface/Time.h"
//#include "CondCore/CondDB/interface/IOVProxy.h"
//#include "CondCore/CondDB/interface/Session.h"

namespace cond {

  namespace db {
    // TO BE CHANGED AFTER THE TRANSITION
    using Session = cond::ora_wrapper::Session;
    using IOVProxy = cond::ora_wrapper::IOVProxy;

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
      
      // reload the iov return true if size has changed
      
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
      
      explicit PayloadProxy( Session& session ) :
	BasePayloadProxy( session ) {}
      
      PayloadProxy( Session& session, const std::string& tag ) :
	BasePayloadProxy(session, tag ) {}
      
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
