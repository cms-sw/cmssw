#include "CondCore/DBCommon/interface/Cipher.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include <string.h>
// blowfish encryption
#include "blowfish.h"
// GNU base 64 encoding
#include "base64.h"

cond::Cipher::Cipher( const std::string& key ):
  m_ctx(new BLOWFISH_CTX){
  char* k = const_cast<char*>(key.c_str());
  Blowfish_Init( m_ctx, reinterpret_cast<unsigned char*>(k), key.size());  
}

cond::Cipher::~Cipher(){
  delete m_ctx;
}

#include <iostream>
void cond::Cipher::test(){
  std::string str = "b";
  char* enc = 0;
  size_t encsize = base64_encode_alloc( str.c_str(), str.size(), &enc );
  //char* enc = new char[str.size()*2+1]; 
  //base64::encoder b64encoder( str.size() );
  //int sz =  b64encoder.encode( str.c_str(), str.size(), enc );
  std::cout <<"## TEST enc = "<<enc<<" sz="<<encsize<<std::endl;
  char* dec = 0;
  size_t decsize = 0;
  //char* dec = new char[sz];
  //base64::decoder b64decoder( sz );
  //int sz2 = b64decoder.decode( enc ,sz, dec );
  if( base64_decode_alloc( enc, encsize, &dec, &decsize ) ){
    std::string strout(dec, decsize);
    std::cout <<"## TEST dec = ["<<strout<<"] sz="<<decsize<<std::endl; 
  } else {
    std::cout <<"### NOT POSSIBLE DECODE!"<<std::endl;
  }
}

std::string cond::Cipher::encrypt( const std::string& input ){
  uInt32 L, R;
  size_t isz = input.size();
  unsigned int j = sizeof(uInt32);

  const char* inputStr = input.c_str();

  unsigned int osz=0;
  for ( unsigned int i=0; i < isz; i+=(j*2)){
    osz = i+2*j;
  }
  char *out = new char[osz];
  memset(out, 0, osz);

  for (unsigned int i=0; i < isz; i+=(j*2)) {
    L = R = 0;
    unsigned int nl = 0;
    unsigned int nr = 0;
    if( isz > i+j ){
      nl = j;
      if( isz > i+2*j ){
	nr = j;
      } else { 
	nr = isz-i-j+1;
      }
    } else {
      nl = isz-i+1;
      nr = 0;
    }
    if(nl) memcpy(&L, inputStr+i, nl);
    if(nr) memcpy(&R, inputStr+i+j, nr);
    Blowfish_Encrypt(m_ctx, &L, &R);
    memcpy(out+i, &L, j);
    memcpy(out+i+j, &R, j);
  }
  
  char* b64out = 0;
  size_t b64size = base64_encode_alloc( out, osz, &b64out );
  std::string ret( b64out, b64size );
  delete [] out;
  free (b64out);
  return ret;
}

std::string cond::Cipher::decrypt( const std::string& input ){
  char* dec = 0;
  size_t decsize = 0;
  if( !base64_decode_alloc( input.c_str(), input.size(), &dec, &decsize ) ){
    throwException("Input provided is not a valid base64 string.","Cipher::decrypt");
  }

  uInt32 L, R;
  unsigned int j = sizeof(uInt32);

  unsigned int isz=0;
  for ( unsigned int i=0; i < decsize; i+=(j*2)){
    isz = i+2*j;
  }
  char *out = new char[isz];
  memset(out, 0, isz);

  for (unsigned int i=0; i < decsize; i+=(j*2)) {
    L = R = 0;
    unsigned int nl = 0;
    unsigned int nr = 0;
    if( isz > i+j ){
      nl = j;
      if( isz > i+2*j ){
	nr = j;
      } else { 
	nr = isz-i-j+1;
      }
    } else {
      nl = isz-i+1;
      nr = 0;
    }
    if(nl) memcpy(&L, dec+i, nl);
    if(nr) memcpy(&R, dec+i+j, nr);
    Blowfish_Decrypt(m_ctx, &L, &R);
    memcpy(out+i, &L, j);
    memcpy(out+i+j, &R, j);
  }
  std::string ret(out);
  free (dec);
  delete [] out;
  return ret;
}

void cond::Cipher::bencrypt( const std::string& input, 
			     std::ostream& outstr ){
  uInt32 L, R;
  size_t isz = input.size();
  unsigned int j = sizeof(uInt32);

  const char* inputStr = input.c_str();

  unsigned int osz=0;
  for ( unsigned int i=0; i < isz; i+=(j*2)){
    osz = i+2*j;
  }
  char *out = new char[osz];
  memset(out, 0, osz);

  for (unsigned int i=0; i < isz; i+=(j*2)) {
    L = R = 0;
    unsigned int nl = 0;
    unsigned int nr = 0;
    if( isz > i+j ){
      nl = j;
      if( isz > i+2*j ){
	nr = j;
      } else { 
	nr = isz-i-j+1;
      }
    } else {
      nl = isz-i+1;
      nr = 0;
    }
    //size_t ptr1 = i;
    //size_t ptr2 = i+j;
    //
    //std::cout <<"Using elements: 0="<<ptr1<<" 1="<<ptr2<<std::endl;
    if(nl) memcpy(&L, inputStr+i, nl);
    if(nr) memcpy(&R, inputStr+i+j, nr);
    Blowfish_Encrypt(m_ctx, &L, &R);
    memcpy(out+i, &L, j);
    memcpy(out+i+j, &R, j);
  }
  outstr.write( out, osz );
}

std::string cond::Cipher::bdecrypt( std::istream& input ){
  char dec[1000];
  size_t decsize = 0;
  while (input.good()){
    dec[decsize] = input.get();  
    decsize++;
  }
  //std::cout <<"Input size="<<decsize<<std::endl;
  uInt32 L, R;
  unsigned int j = sizeof(uInt32);

  unsigned int isz=0;
  for ( unsigned int i=0; i < decsize; i+=(j*2)){
    isz = i+2*j;
  }
  char *out = new char[isz];
  memset(out, 0, isz);

  for (unsigned int i=0; i < decsize; i+=(j*2)) {
    L = R = 0;
    unsigned int nl = 0;
    unsigned int nr = 0;
    if( isz > i+j ){
      nl = j;
      if( isz > i+2*j ){
	nr = j;
      } else { 
	nr = isz-i-j+1;
      }
    } else {
      nl = isz-i+1;
      nr = 0;
    }
    //size_t ptr1 = i;
    //size_t ptr2 = i+j;
    //std::cout <<"Using elements: 0="<<ptr1<<" 1="<<ptr2<<" nl="<<nl<<" nr="<<nr<<std::endl;
    if(nl) memcpy(&L, dec+i, nl);
    if(nr) memcpy(&R, dec+i+j, nr);
    Blowfish_Decrypt(m_ctx, &L, &R);
    memcpy(out+i, &L, j);
    memcpy(out+i+j, &R, j);
  }
  std::string ret(out);
  delete [] out;
  return ret;
}
