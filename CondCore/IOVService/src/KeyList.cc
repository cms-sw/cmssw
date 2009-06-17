#include "CondCore/IOVService/interface/KeyList.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "DataSvc/RefException.h"



namespace cond {

  KeyList::KeyList(IOVKeysDescription const * idescr) : m_description(idescr){}
  
  void KeyList::load(std::vector<unsigned long long> const & keys) {
    m_sequence.db().start(true);
    m_data.resize(keys.size());
    for (int i=0; i<keys.size(); i++) {
      if (keys[i]!=0) {
	IOVSequence::const_iterator p = m_sequence.iov().find(keys[i]);
	pool::Ref<Wrapper> ref(&(m_sequence.db().poolDataSvc()),(*p).wrapperToken());
	m_data[i].copyShallow(ref);
      } else m_data[i].clear();
    }
    m_sequence.db().commit();
  }


  BaseKeyed const * KeyList::elem(int n) const {
    if (!m_data[n]) return 0;
    return &(*m_data[n]).data();
  }
  





}
