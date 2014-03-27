#include "CondCore/CondDB/interface/KeyList.h"
#include "CondCore/CondDB/interface/Session.h"

namespace cond {

  namespace persistency {

    void KeyList::init(  IOVProxy iovProxy ){
      m_proxy = iovProxy;
      m_data.clear();
      m_objects.clear();
    }

    void KeyList::load( const std::vector<unsigned long long>& keys ){
      std::shared_ptr<SessionImpl> simpl = m_proxy.session();
      if( !simpl.get() ) cond::throwException("The KeyList has not been initialized.","KeyList::load");
      Session s( simpl );
      s.transaction().start( true );
      m_isOra = s.isOraSession();
      m_data.clear();
      m_objects.resize(keys.size());
      for (size_t i=0; i<keys.size(); ++i) {
	m_objects[i].reset();
	if (keys[i]!=0) {
	  auto p = m_proxy.find(keys[i]);
	  if ( p!= m_proxy.end()) {
	    auto item = m_data.insert( std::make_pair( i, std::make_pair("",cond::Binary()) ) );
	    if( ! s.fetchPayloadData( (*p).payloadId, item.first->second.first, item.first->second.second ) )
	      cond::throwException("The Iov contains a broken payload reference.","KeyList::load");    
	  }
	}
      }
      s.transaction().commit();
    }
  }
}


