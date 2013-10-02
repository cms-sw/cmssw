#include "CondCore/CondDB/interface/KeyList.h"
#include "CondCore/CondDB/interface/IOVProxy.h"

new_impl::KeyList::KeyList( Session& session ):
  m_session( session ),
  m_data(),
  m_objects(){
}

void new_impl::KeyList::load(const std::string& tag, const std::vector<unsigned long long>& keys){
  m_session.transaction().start( true );
  new_impl::IOVProxy proxy = m_session.readIov( tag );
  m_data.clear();
  m_objects.resize(keys.size());
  for (size_t i=0; i<keys.size(); ++i) {
    m_objects[i].reset();
    if (keys[i]!=0) {
      auto p = proxy.find(keys[i]);
      if ( p!= proxy.end()) {
        auto item = m_data.insert( std::make_pair( i, std::make_pair("",conddb::Binary()) ) );
	if( ! m_session.fetchPayloadData( (*p).payloadId, item.first->second.first, item.first->second.second ) )
	  conddb::throwException("The Iov contains a broken payload reference.","KeyList::load");    
      }
    }
  }
  m_session.transaction().commit();
}


