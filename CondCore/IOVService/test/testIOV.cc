#include "CondCore/IOVService/src/IOV.h"

#include<iostream>

int main() {

  // old iov
  cond::IOV iov;
  
  iov.add(10,"a");
  iov.add(20,"b");
  if (iov.add(30,"c")!=2) std::cerr << "error pos" << std::endl;
  iov.add(40,"d");

  cond::IOV::Container const & v = iov.iov;

  if (v.size()!=4) std::cerr << "error size" << std::endl;
  
  if (iov.find(0)!=v.begin()) std::cerr << "error find 0" << std::endl;
  if (iov.find(5)!=v.begin()) std::cerr << "error find 5" << std::endl;
  if (iov.find(10)!=v.begin()) std::cerr << "error find 10" << std::endl;
  if (iov.find(25)!=(v.begin()+2)) std::cerr << "error find 25" << std::endl;
  if (iov.find(45)!=v.end()) std::cerr << "error find 45" << std::endl;


  return 0;

}
