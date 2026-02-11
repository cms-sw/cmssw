#include "FWCore/Utilities/interface/OftenEmptyCString.h"
#include <cstring>
#include <utility>

namespace edm {
  OftenEmptyCString::OftenEmptyCString(const char* iValue) {
    if (iValue == nullptr or iValue[0] == '\0') {
      m_value = emptyString();
    } else {
      auto l = strlen(iValue);
      auto temp = new char[l + 1];
      strncpy(temp, iValue, l + 1);
      m_value = temp;
    }
  }
  void OftenEmptyCString::deleteIfNotEmpty() {
    if (m_value != emptyString()) {
      delete[] m_value;
    }
  }
  OftenEmptyCString::~OftenEmptyCString() { deleteIfNotEmpty(); }

  OftenEmptyCString::OftenEmptyCString(OftenEmptyCString const& iOther) : OftenEmptyCString(iOther.m_value) {}
  OftenEmptyCString::OftenEmptyCString(OftenEmptyCString&& iOther) noexcept : m_value(iOther.m_value) {
    iOther.m_value = nullptr;
  }
  OftenEmptyCString& OftenEmptyCString::operator=(OftenEmptyCString const& iOther) {
    if (iOther.m_value != m_value) {
      OftenEmptyCString temp{iOther};
      *this = std::move(temp);
    }
    return *this;
  }
  OftenEmptyCString& OftenEmptyCString::operator=(OftenEmptyCString&& iOther) noexcept {
    if (iOther.m_value != m_value) {
      deleteIfNotEmpty();
      m_value = iOther.m_value;
      iOther.m_value = nullptr;
    }
    return *this;
  }

  const char* OftenEmptyCString::emptyString() {
    constexpr static const char* s_empty = "";
    return s_empty;
  }

}  // namespace edm