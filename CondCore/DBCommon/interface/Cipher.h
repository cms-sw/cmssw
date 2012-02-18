#ifndef INCLUDE_COND_CIPHER_H
#define INCLUDE_COND_CIPHER_H

#include <iostream>
#include <string>

struct BLOWFISH_CTX;

namespace cond {

  class Cipher {
  public:

    explicit Cipher( const std::string& key );

    ~Cipher();

    //std::pair<char*,size_t> encrypt( const std::string& input );

    //std::string decrypt( const char* input,size_t sz );
    std::string encrypt( const std::string& input );

    std::string decrypt( const std::string& input );

    void bencrypt( const std::string& input, std::ostream& out );

    std::string bdecrypt( std::istream& input );

    void test();

  private:
    
    BLOWFISH_CTX* m_ctx;
  };

}

#endif

