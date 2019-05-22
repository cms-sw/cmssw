#include "CondFormats/Common/interface/SmallWORMDict.h"
#include <string>
#include <functional>
#include <numeric>

namespace cond {
  using namespace std::placeholders;

  SmallWORMDict::SmallWORMDict() {}
  SmallWORMDict::~SmallWORMDict() {}

  SmallWORMDict::SmallWORMDict(std::vector<std::string> const& idict)
      : m_data(std::accumulate(idict.begin(), idict.end(), 0, [](int a, std::string b) { return a + b.size(); })),
        m_index(idict.size(), 1) {
    // sort (use index)
    m_index[0] = 0;
    std::partial_sum(m_index.begin(), m_index.end(), m_index.begin());
    std::sort(m_index.begin(), m_index.end(), [&idict](unsigned int a, unsigned int b) {
      return std::less<std::string>()(idict[a], idict[b]);
    });

    //copy
    std::vector<char>::iterator p = m_data.begin();
    for (size_t j = 0; j < m_index.size(); j++) {
      size_t i = m_index[j];
      p = std::copy(idict[i].begin(), idict[i].end(), p);
      m_index[j] = p - m_data.begin();
    }
  }

  struct LessFrame {
    bool operator()(SmallWORMDict::Frame const& rh, SmallWORMDict::Frame const& lh) const {
      return std::lexicographical_compare(rh.b, rh.b + rh.l, lh.b, lh.b + lh.l);
    }
  };

  size_t SmallWORMDict::index(std::string const& s) const { return (*find(s)).ind; }

  size_t SmallWORMDict::index(char const* s) const { return (*find(s)).ind; }

  SmallWORMDict::const_iterator SmallWORMDict::find(std::string const& s) const {
    Frame sp(&s[0], s.size(), 0);
    return std::lower_bound(begin(), end(), sp, LessFrame());
  }

  SmallWORMDict::const_iterator SmallWORMDict::find(char const* s) const {
    Frame sp(s, ::strlen(s), 0);
    return std::lower_bound(begin(), end(), sp, LessFrame());
  }

  size_t SmallWORMDict::size() const { return m_index.size(); }

}  // namespace cond
