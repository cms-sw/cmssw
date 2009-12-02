#include <iostream>
#include <string>
#include <sstream>
int main(){
  std::string s="000121998";
  std::istringstream myStream(s);
  unsigned int i;
  if(myStream>>i){
    std::cout<<"i "<<i<<std::endl;
  }else{
    std::cout<<"conversion error"<<std::endl;
  }
  int t=21;
  std::stringstream ss;
  ss.width(3);
  ss.fill('0');
  ss<<t;
  std::cout<<ss.str()<<std::endl;
  return 0;
}
