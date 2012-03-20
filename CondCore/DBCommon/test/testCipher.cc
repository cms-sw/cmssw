#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/Cipher.h"
#include <iostream>

int main(){
  cond::KeyGenerator gen;
  for( unsigned int i = 0; i<200; i++ ){
    
    std::string word = gen.makeWithRandomSize( 200 );
    std::string k = gen.makeWithRandomSize( 100 );
    cond::Cipher cipher0( k );
    std::string encr = cipher0.b64encrypt( word );
    cond::Cipher cipher1( k );
    std::string decr = cipher1.b64decrypt( encr );
    if( word != decr ){
      std::cout <<"##### Error: encoded ["<<word<<"] (size="<<word.size()<<") with key="<<k<<"; decoded=["<<decr<<"] (size="<<decr.size()<<") "<<std::endl;
    } else {
      std::cout <<"TEST OK: encoded size "<<word.size()<<" with key size="<<k.size()<<std::endl;
    }
  }
}
