#ifndef Common_MD5Hash_h
#define Common_MD5Hash_h

/*----------------------------------------------------------------------
  
MD5Hash:

$Id: MD5Hash.h,v 1.1.2.3 2006/06/27 21:05:17 paterno Exp $
----------------------------------------------------------------------*/
namespace edm {
  struct MD5Hash {
    static int const size = 16;
    MD5Hash() {}
    ~MD5Hash() {}
    unsigned char value_[size];
  };

  bool operator<(MD5Hash const& a, MD5Hash const& b);

  bool operator==(MD5Hash const& a, MD5Hash const& b);

  inline
  bool operator!=(MD5Hash const& a, MD5Hash const& b) {
    return !(a == b);
  }
}
#endif
