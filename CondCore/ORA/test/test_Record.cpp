#include "CondCore/ORA/interface/Record.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>


#ifndef __APPLE__
#include<malloc.h>
#endif

#include<cstdlib>
#include "cxxabi.h"

namespace {
  inline void printMem(char const * title) {
    std::cout << "\n--- " << title <<" ---"<< std::endl;
#ifdef __APPLE__
    std::cout << "not supported" << std::endl;
    abort();
#else
    struct mallinfo mi;
    mi  = mallinfo();
    int * mm = (int*)(&mi);
    for(int i=0;i<10;i++) std::cout << mm[i] << ", ";
    std::cout << std::endl;
    std::cout << "mmap/arena-used/arena-free " << mi.hblkhd << " " << mi.uordblks << " " << mi.fordblks << std::endl;
    std::cout << "mmap/arena-used/arena-free " << mm[4] << " " << mm[7] << " " << mm[8] << std::endl;
    std::cout << std::endl;
    malloc_stats();
#endif
  }
  
  
  void checkmem(char const * title){
    std::cout << "\n--- " << title <<" ---"<< std::endl;
#ifdef __APPLE__
    std::cout << "malloc_stats not supported\n" << std::endl;
    abort();
#else
    malloc_stats();
#endif
  }

}


char const * typeName(std::type_info const & type) {
  int    status = 0;
  return __cxxabiv1::__cxa_demangle(type.name(), 0, 0, &status);
}

using namespace ora;

void testRecordFeatures() {
  RecordSpec specs; 
  RecordSpec bindspecs; 

  std::ostringstream oss;
  std::string f("f_");
  for (int i=0;i<100; ++i) {
    oss.str("");
    oss << i;
    specs.add(f+oss.str(), typeid(float));
    bindspecs.add(f+oss.str(), typeid(float*));
  }

  Record record(specs);
  Record brecord(bindspecs);

  std::cout << record.size() << std::endl;

  std::cout << record.index("f") << std::endl;

  std::cout << record.index("f_50") << std::endl;

  float ff = 3.14;
  record.set(50,&ff);

  std::cout << *reinterpret_cast<float const*>(record.get(50)) << std::endl;
  std::cout << record.data<float>(50) << std::endl;
  record.data<float>(50) = 6.28;
  std::cout << *reinterpret_cast<float const*>(record.get(50)) << std::endl;
  std::cout << record.data<float>(50) << std::endl;


  float * pf = &ff;
  brecord.set(50, &pf);

  float * const p2 =  *reinterpret_cast<float * const*>(brecord.get(50));

  std::cout << *p2 << std::endl;
  std::cout << brecord.data<float>(50) << std::endl;

  *p2 = 6.28;
  
  std::cout << ff << std::endl;
  brecord.data<float>(50) = 12.56;
  std::cout << ff << std::endl;

  std::cout << "verify swap" << std::endl;
  
  Record record2;
  std::cout << "before swap" << record2.size() << " " << record.size() << std::endl;
  swap(record, record2);
  std::cout << "after swap" << record2.size() << " " << record.size() << std::endl;
  std::cout << record2.index("f_50")  << " " << record.index("f_50") << std::endl;
  std::cout << record2.data<float>(50) << std::endl;


}

// now the real stuff...
void testRecord(std::vector<float> const & v, std::vector<float> & v2) {
  checkmem("before specs");

  RecordSpec specs;
  {
    std::ostringstream oss;
    std::string f("f_");
    for (int i=0;i<100; ++i) {
      oss.str("");
      oss << i;
      specs.add(f+oss.str(), typeid(float));
    }
  }

  checkmem("after specs");
  Record record(specs);
  checkmem("after record");

  for (int i=0;i<100; ++i)
    record.data<float>(i) = v[i];

  checkmem("after assign");

  for (int i=0;i<100; ++i)
    v2[i]= record.data<float>(i);

   checkmem("after recover");
 
}

#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeListSpecification.h"

void testAttributeList(std::vector<float> const & v, std::vector<float> & v2) {
  checkmem("before AttributeListSpecs");

  coral::AttributeListSpecification * specs = new coral::AttributeListSpecification();

  {
    std::ostringstream oss;
    std::string f("f_");
    for (int i=0;i<100; ++i) {
      oss.str("");
      oss << i;
      specs->extend(f+oss.str(), typeid(float));
    }
  }

  checkmem("after  AttributeSpecs");
  
  coral::AttributeList record(*specs,true);
  specs->release();
  checkmem("after  AttributeList");

  for (int i=0;i<100; ++i)
    record[i].data<float>() = v[i];

  checkmem("after assign");

  for (int i=0;i<100; ++i)
    v2[i]= record[i].data<float>();

   checkmem("after recover");
 
}

int main() {

  checkmem("start");
  {
    std::vector<float> v(100,0.);
    std::vector<float> v2(100,0.);
    std::vector<float> v3(100,0.);
    for (int i=0;i<100; ++i)
      v[i] = float(i)+0.01*float(i);
    checkmem("after vector");
    
    testRecord(v,v2);
    if (v!=v2) std::cout << "error in Record" << std::endl;
    checkmem("after Record done");

    testAttributeList(v,v3);
    if (v!=v3) std::cout << "error in AttributeList" << std::endl;
    checkmem("after AttributeList done");
  }

  checkmem("Before Features");
  testRecordFeatures();
  checkmem("end");

  return 0;

}
