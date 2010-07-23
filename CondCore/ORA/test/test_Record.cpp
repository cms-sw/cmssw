#include "CondCore/ORA/interface/Record.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include<malloc.h>
#include<cstdlib>

namespace {
  void printMem(char const * title) {
	std::cout << "\n--- " << title <<" ---"<< std::endl;
	struct mallinfo mi;
	mi  = mallinfo();
	int * mm = (int*)(&mi);
	for(int i=0;i<10;i++) std::cout << mm[i] << ", ";
	std::cout << std::endl;
        std::cout << "mmap/arena-used/arena-free " << mi.hblkhd << " " << mi.uordblks << " " << mi.fordblks << std::endl;
        std::cout << "mmap/arena-used/arena-free " << mm[4] << " " << mm[7] << " " << mm[8] << std::endl;
        std::cout << std::endl;
        malloc_stats();
  }


  void checkmem(char const * title){
       std::cout << "\n--- " << title <<" ---"<< std::endl;
       malloc_stats();
  }

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
}

// now the real stuff...
void testRecord(std::vector<float> const & v, std::vector<float> & v2) {
  checkmem("before specs");

  RecordSpec specs;
  std::ostringstream oss;
  std::string f("f_");
  for (int i=0;i<100; ++i) {
    oss.str("");
    oss << i;
    specs.add(f+oss.str(), typeid(float));
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


int main() {

  checkmem("start");
  std::vector<float> v(100,0.);
  std::vector<float> v2(100,0.);
  for (int i=0;i<100; ++i)
    v[i] = float(i)+0.01*float(i);
  checkmem("after vector");
  testRecord(v,v2);
  if (v!=v2) std::cout << "error in Record" << std::endl;
  checkmem("after Record done");



  testRecordFeatures();

  return 0;

}
