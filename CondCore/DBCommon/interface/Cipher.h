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

    size_t encrypt( const std::string& input, unsigned char*& output );

    std::string decrypt( const unsigned char* input, size_t inputSize );

    std::string b64encrypt( const std::string& input );

    std::string b64decrypt( const std::string& input );

  private:

    size_t bf_process_alloc( const unsigned char* input, size_t input_size, unsigned char*& output, bool decrypt=false );
    
  private:

    BLOWFISH_CTX* m_ctx;
  };

}

#endif

