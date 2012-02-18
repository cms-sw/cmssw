#include "CondCore/DBCommon/interface/Cipher.h"
#include <iostream>

int main(){
  std::string key("ABCDdfyfyfugugugudddrsfgrdfddfffgfgfgglsdsdsdsedgueduehciljhcowjcodcjwdcjdojjdedoedqpwns]ppwlmmzpwmsad");
  std::string mystring("0123456789abcdefghijklmnopqrstuvwxyzABCDEFG");
  cond::Cipher cipher( key );
  std::string en = cipher.encrypt( mystring );
  unsigned char A = '/';
  unsigned char a_ = '9';
  unsigned char at = '8';
  std::cout <<"### slash="<<(int)A<<" 9="<<(int)a_<<" +="<<(int)at<<std::endl;
  std::cout <<"### String=\""<<mystring<<"\" en="<<en<<std::endl;
  std::string de = cipher.decrypt( en );
  std::cout <<"### decr=\""<<de<<"\" "<<std::endl;  
  cipher.test();
  //delete [] en;
  return 0;
}
