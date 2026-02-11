#ifndef FWCore_Utilities_OftenEmptyCString_h
#define FWCore_Utilities_OftenEmptyCString_h

/* Class is optimized to not require any additional memory
if the string passes is empty. A nullptr will be replaced with
the empty string as well.
*/

namespace edm {
  class OftenEmptyCString {
  public:
    OftenEmptyCString() : m_value(emptyString()) {}
    ~OftenEmptyCString();
    explicit OftenEmptyCString(const char*);
    OftenEmptyCString(OftenEmptyCString const&);
    OftenEmptyCString(OftenEmptyCString&&) noexcept;
    OftenEmptyCString& operator=(OftenEmptyCString const&);
    OftenEmptyCString& operator=(OftenEmptyCString&&) noexcept;

    const char* c_str() const noexcept { return m_value; }

  private:
    static const char* emptyString();
    void deleteIfNotEmpty();
    char const* m_value;
  };
}  // namespace edm

#endif