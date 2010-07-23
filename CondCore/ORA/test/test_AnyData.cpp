#include "CondCore/ORA/src/AnyData.h"
#include "CondCore/ORA/src/RecordDetails.h"
#include <iostream>
#include <string>
#include <vector>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits.hpp>
using namespace ora;

template<typename T>
void go(TypeHandler & h, AnyData & ad, void * p) {
  std::cout << h.type->name() << 
    (  h.isPointer() ? " pointer " : " ") <<  (void *)(p) << std::endl;
  std::cout <<  *reinterpret_cast<T*>(p) << std::endl;
  ad.p =0;
  std::cout << (void *)(ad.p) << std::endl;
  h.create(ad);
  std::cout << (void *)(ad.p) << std::endl;
  h.set(ad,p);
  std::cout << (void *)(ad.p) << std::endl;
  void const * pp = h.get(ad);
  std::cout << (void *)(ad.p) << std::endl;
  std::cout << (void *)(&ad.p) << std::endl;
  std::cout << (void *)(pp) << std::endl;
  std::cout <<  *reinterpret_cast<T const*>(pp) << std::endl;
  pp = h.address(ad);
  typedef typename  boost::add_pointer< typename boost::add_const< typename boost::remove_pointer<T>::type >::type >::type TCP;
  std::cout << "Value is " << *reinterpret_cast<TCP>(pp) << std::endl;;

  h.destroy(ad);
  std::cout << (void *)(ad.p) << std::endl;

}

int main() {


  AnyData ad;
  AnyData aa;
  std::cout << sizeof(AnyData) << std::endl;
  std::cout << sizeof(std::string) << std::endl;
  std::cout << sizeof(std::vector<double>) << std::endl;
  std::cout << sizeof(long double) << std::endl;
 
  ad.data<int>() = 3;
  std::cout << ad.data<int>() << std::endl;

  aa.data<AnyData*>() = &ad;
  AnyData * ap =  aa.data<AnyData*>();
  std::cout << ap->data<int>() << std::endl;


  std::cout << "test AnyTypeHandler" << std::endl;
  
  AnyTypeHandler<int> ih;
  AnyTypeHandler<int*> iph;
  AnyTypeHandler<std::string> ssh;
  AnyTypeHandler<long double> ldh;
  int i3 = 3;
  int * p3 = &i3;
  long double ld = 3.1415;
  std::string ss("Hello World");
  go<int>(ih,ad, &i3);
  go<int*>(iph,ad, &p3);
  go<std::string>(ssh,ad,&ss);
  go<long double>(ldh,ad, &ld);


  AllKnowTypeHandlers  allKnowTypeHandlers;
  TypeHandler const * th =  allKnowTypeHandlers(typeid(int));
  std::cout << typeid(*th).name() << std::endl;

  th =  allKnowTypeHandlers(typeid(std::vector<double>));
  std::cout << th << std::endl;
  

  return 0;
}
