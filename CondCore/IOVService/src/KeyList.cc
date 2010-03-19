#include "CondCore/IOVService/interface/KeyList.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"



namespace cond {

  KeyList::KeyList(IOVKeysDescription const * idescr) : m_description(idescr){}
  
  void KeyList::load(std::vector<unsigned long long> const & keys) {
    m_sequence.db().transaction().start(true);
    m_data.resize(keys.size());
    for (size_t i=0; i<keys.size(); ++i) {
      m_data[i].clear();
      if (keys[i]!=0) {
        IOVSequence::const_iterator p = m_sequence.iov().findSince(keys[i]);
	if (p!=m_sequence.iovs().end()) { 
	  pool::Ref<Wrapper> ref = m_sequence.db().getTypedObject<Wrapper>( (*p).wrapperToken() );
	  m_data[i].copyShallow(ref);
	  m_data[i]->loadAll();
	}
      }
    }
    m_sequence.db().transaction().commit();
  }


  BaseKeyed const * KeyList::elem(int n) const {
    if (!m_data[n]) return 0;
    return &(*m_data[n]).data();
  }
  

}
