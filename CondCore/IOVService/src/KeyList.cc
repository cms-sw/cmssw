#include "CondCore/IOVService/interface/KeyList.h"



namespace cond {

  KeyList::KeyList(IOVKeysDescription const * idescr) : m_description(idescr){}
  
  void KeyList::load(std::vector<unsigned long long> const & keys) {
    m_sequence.db().start(true);
    for (int i=0; i<names.size(),i++) {
      IOVSequence::const_iterator p = m_sequence.iov().find(keys[i]);
      pool::Ref<DataWrapper> ref(&(m_sequence.db().poolDataSvc()),(*p).wrapperToken());
      m_data[i].copyShallow(ref);
    }
    m_sequence.db().commit();
  }








}
