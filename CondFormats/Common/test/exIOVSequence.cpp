#include "CondFormats/Common/interface/IOVSequence.h"

#include<iostream>

int main() {


  cond::IOVSequence iov;
  
  iov.add(10,"a","class0");
  iov.add(20,"b","class0");
  if (iov.add(30,"c","class0")!=2) std::cerr << "error pos" << std::endl;
  iov.add(40,"e","class0");

  {
    cond::IOVSequence::Container const & v = iov.iovs();

    if (v.size()!=4) std::cerr << "error size" << std::endl;
    
    if (iov.find(0)!=v.end()) std::cerr << "error find 0" << std::endl;
    if (iov.find(5)!=v.end()) std::cerr << "error find 5" << std::endl;
    if (iov.find(10)!=v.begin()) std::cerr << "error find 10" << std::endl;
    if (iov.find(15)!=(v.begin())) std::cerr << "error find 15" << std::endl;
    if (iov.find(25)!=(v.begin()+1)) std::cerr << "error find 25" << std::endl;
    if (iov.find(35)!=(v.begin()+2)) std::cerr << "error find 35" << std::endl;
    if (iov.find(36)!=(v.begin()+2)) std::cerr << "error find 36" << std::endl;
    if (iov.find(40)!=v.end()-1) std::cerr << "error find 40" << std::endl;
    if (iov.find(45)!=v.end()-1) std::cerr << "error find 45" << std::endl;
    if (iov.findSince(12)!=v.end()) std::cerr << "error findSince 12" << std::endl;
    if (iov.findSince(20)!=(v.begin()+1)) std::cerr << "error findSince 20" << std::endl;


    if (iov.add(50,"f","class0")!=4)  std::cerr << "error in add"  << std::endl;
    if (iov.find(45)!=v.end()-2) std::cerr << "error find 45" << std::endl;
    if(iov.truncate()!=3)  std::cerr << "error in truncation"  << std::endl;
    if (iov.find(45)!=v.end()-1) std::cerr << "error find 45" << std::endl;

    if (iov.add(50,"f","class0")!=4)  std::cerr << "error in add"  << std::endl;
    if (iov.find(45)!=v.end()-2) std::cerr << "error find 45" << std::endl;
    if(iov.truncate()!=3)  std::cerr << "error in truncation"  << std::endl;
    if (iov.find(45)!=v.end()-1) std::cerr << "error find 45" << std::endl;

 

  }

  if(iov.notOrdered()) std::cerr << "error notOrdered" << std::endl;

  iov.add(35,"d","class0");
  if(!iov.notOrdered()) std::cerr << "error not notOrdered" << std::endl;

  { 

    cond::IOVSequence::Container const & v = iov.iovs();

    if (v.size()!=5) std::cerr << "error size" << std::endl;
    if (iov.find(25)!=(v.begin()+1)) std::cerr << "error find 25" << std::endl;
    if (iov.find(35)!=(v.begin()+3)) std::cerr << "error find 35" << std::endl;
    if (iov.find(36)!=(v.begin()+3)) std::cerr << "error find 36" << std::endl;


    if(!iov.notOrdered()) std::cerr << "error not notOrdered" << std::endl;


  }



  return 0;

}
