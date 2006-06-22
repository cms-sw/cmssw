#ifndef FWCORE_FWUTILITIES_DIGEST_INC
#define FWCORE_FWUTILITIES_DIGEST_INC

#include <string>
#include "md5.h"


namespace cms
{
  struct MD5Result
  {
    unsigned char bytes[16];
    std::string toString() const;
  };


  class Digest
  {
  public:
    Digest();
    explicit Digest(std::string const& s);

    void append(std::string const& s);

    MD5Result digest() const;

  private:
    mutable md5_state_t state_;
  };
}

#endif
