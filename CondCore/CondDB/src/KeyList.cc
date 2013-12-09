#include "CondCore/CondDB/interface/KeyList.h"
#include "CondCore/CondDB/interface/IOVProxy.h"

namespace cond {

  namespace persistency {

    KeyList::KeyList( Session& session ):
      m_tag(""),
      m_session( session ),
      m_data(),
      m_objects(){
    }

    void KeyList::init( const std::string& tag ){
      m_tag = tag;
      m_data.clear();
      m_objects.clear();
    }

    void KeyList::load( const std::vector<unsigned long long>& keys ){
      m_session.transaction().start( true );
      IOVProxy proxy = m_session.readIov( m_tag );
      m_data.clear();
      m_objects.resize(keys.size());
      for (size_t i=0; i<keys.size(); ++i) {
	m_objects[i].reset();
	if (keys[i]!=0) {
	  auto p = proxy.find(keys[i]);
	  if ( p!= proxy.end()) {
	    auto item = m_data.insert( std::make_pair( i, std::make_pair("",cond::Binary()) ) );
	    if( ! m_session.fetchPayloadData( (*p).payloadId, item.first->second.first, item.first->second.second ) )
	      throwException("The Iov contains a broken payload reference.","KeyList::load");    
	  }
	}
      }
      m_session.transaction().commit();
    }

  }
}


