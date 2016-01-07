#include "CondCore/CondDB/interface/Cipher.h"
#include "CondCore/CondDB/interface/Exception.h"
#include <string.h>
// blowfish encryption
#include "blowfish.h"
// GNU base 64 encoding
#include "base64.h"
#include <cassert>

cond::Cipher::Cipher( const std::string& key ):
  m_ctx(new BLOWFISH_CTX){
  char* k = const_cast<char*>(key.c_str());
  Blowfish_Init( m_ctx, reinterpret_cast<unsigned char*>(k), key.size());  
}

cond::Cipher::~Cipher(){
  delete m_ctx;
}

size_t cond::Cipher::bf_process_alloc( const unsigned char* input, 
				       size_t input_size, 
				       unsigned char*& output,
				       bool decrypt ){
  assert(input_size != 0);

  uInt32 L, R;
  unsigned int j = sizeof(uInt32);

  unsigned int output_size=0;

  if( !input_size ) {
    output = 0;
    return 0;
  }

  for ( unsigned int i=0; i < input_size; i+=(j*2)){
    output_size = i+2*j;
  }
  output = (unsigned char*) malloc( output_size );
  memset(output, 0, output_size);

  for (unsigned int i=0; i < input_size; i+=(j*2)) {
    L = R = 0;
    unsigned int nl = 0;
    unsigned int nr = 0;
    if( input_size > i+j ){
      nl = j;
      if( input_size > i+2*j ){
	nr = j;
      } else { 
	nr = input_size-i-j;
      }
    } else {
      nl = input_size-i;
      nr = 0;
    }
    if(nl) memcpy(&L, input+i, nl);
    if(nr) memcpy(&R, input+i+j, nr);
    if( !decrypt ){
      Blowfish_Encrypt(m_ctx, &L, &R);
    } else {
      Blowfish_Decrypt(m_ctx, &L, &R);
    }
    memcpy(output+i, &L, j);
    memcpy(output+i+j, &R, j);
  }
  
  return output_size;
}
    
size_t cond::Cipher::encrypt( const std::string& input, unsigned char*& output ){
  if( input.empty() ) {
    output = 0;
    return 0;
  }
  return bf_process_alloc( reinterpret_cast<const unsigned char*>(input.c_str()), input.size(), output, false );;
}

std::string cond::Cipher::decrypt( const unsigned char* input, size_t inputSize ){
  if( !inputSize ) return ""; 
  unsigned char* out = 0;
  size_t outSize = bf_process_alloc( input, inputSize, out, true );
  size_t i = 0;
  for( i=0;i<outSize; i++ ) {
    if( out[i]==0 ) break;
  }

  char* sout = reinterpret_cast<char*>(out);
  // the output can still contain one or more \0 chars...
  //size_t soutSize = strlen( sout ); 
  size_t soutSize = 0;
  for( soutSize=0; soutSize<outSize; soutSize++) if( out[soutSize]==0 ) break;

  if( soutSize < outSize ){
    outSize = soutSize;
  }

  std::string ret("");
  if( outSize ) 
    ret = std::string( sout, outSize );
  free (out );
  return ret;
}

std::string cond::Cipher::b64encrypt( const std::string& input ){
  if( input.empty() ) return "";
  unsigned char* out = 0;
  size_t outSize = bf_process_alloc( reinterpret_cast<const unsigned char*>(input.c_str()), input.size(), out, false );
  char* b64out = 0;
  size_t b64size = base64_encode_alloc( reinterpret_cast<const char*>(out), outSize, &b64out );
  std::string ret( b64out, b64size );
  free (out);
  free (b64out);
  return ret;
}

std::string cond::Cipher::b64decrypt( const std::string& b64in ){
  if( b64in.empty() ) return "";
  char* input = 0;
  size_t inputSize = 0;
  if( !base64_decode_alloc( b64in.c_str(), b64in.size(), &input, &inputSize ) ){
    throwException("Input provided is not a valid base64 string.","Cipher::b64decrypt");
  }
  std::string ret = decrypt( reinterpret_cast<const unsigned char*>(input), inputSize );
  free (input);
  return ret;
}

