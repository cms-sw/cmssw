#include "CondCore/CondDB/interface/KeyList.h"
#include "CondCore/CondDB/interface/Session.h"

namespace cond {

  namespace persistency {

    void KeyList::init(IOVProxy iovProxy) {
      m_proxy = iovProxy;
      m_data.clear();
      m_keys.clear();
    }

    void KeyList::init(KeyList const& originalKeyList) { init(originalKeyList.m_proxy); }

    void KeyList::setKeys(const std::vector<unsigned long long>& keys) {
      std::shared_ptr<SessionImpl> simpl = m_proxy.session();
      if (!simpl.get())
        cond::throwException("The KeyList has not been initialized.", "KeyList::setKeys");
      Session s(simpl);
      s.transaction().start(true);
      m_keys = keys;
      std::sort(m_keys.begin(), m_keys.end(), std::less<unsigned long long>());
      m_data.clear();
      m_data.resize(keys.size(), std::make_pair("", std::make_pair(cond::Binary(), cond::Binary())));
      for (size_t i = 0; i < m_keys.size(); ++i) {
        if (m_keys[i] != 0) {
          auto p = m_proxy.find(m_keys[i]);
          if (p != m_proxy.end()) {
            auto& item = m_data[i];
            if (!s.fetchPayloadData((*p).payloadId, item.first, item.second.first, item.second.second))
              cond::throwException("The Iov contains a broken payload reference.", "KeyList::setKeys");
          }
        }
      }
      s.transaction().commit();
    }

    std::pair<std::string, std::pair<cond::Binary, cond::Binary> > KeyList::loadFromDB(unsigned long long key) const {
      std::pair<std::string, std::pair<cond::Binary, cond::Binary> > item;
      if (key == 0) {
        return item;
      }
      std::shared_ptr<SessionImpl> simpl = m_proxy.session();
      if (!simpl.get())
        cond::throwException("The KeyList has not been initialized.", "KeyList::loadFromDB");
      Session s(simpl);
      s.transaction().start(true);
      auto p = m_proxy.find(key);
      if (p != m_proxy.end()) {
        if (!s.fetchPayloadData((*p).payloadId, item.first, item.second.first, item.second.second))
          cond::throwException("The Iov contains a broken payload reference.", "KeyList::loadFromDB");
      } else {
        throwException("Payload for key " + std::to_string(key) + " has not been found.", "KeyList::loadFromDB");
      }
      s.transaction().commit();
      return item;
    }

  }  // namespace persistency
}  // namespace cond
