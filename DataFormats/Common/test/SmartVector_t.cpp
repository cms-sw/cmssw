#include "DataFormats/Common/interface/SmartVector.h"

#include<iostream>
#include<cassert>

template<typename T>
int go() {

 using Vector = std::vector<T>;
 using Array = std::array<T,sizeof(Vector)/sizeof(T)>;
 using Variant = SmartVector<T>;


 std::cout << sizeof(Vector) <<' '<< sizeof(Array) <<' '<< sizeof(Variant) << std::endl;

 Variant v;

 T data[128];
 for (int i=0; i<128; ++i) data[i]=i;

 T i=0;
 Variant va(data,data+5);
 assert(5==va.size());
 assert(5==va.end()-va.begin());
 assert(std::get_if<Array>(&va.container()));
 i=0; for (auto c : va) assert(c==i++);
 Variant vb; vb.initialize(data,data+24);
 assert(24==vb.size());
 assert(24==vb.end()-vb.begin());
 assert(std::get_if<Vector>(&vb.container()));
 i=0; for (auto c : vb) assert(c==i++);
 Variant vv; vv.extend(data,data+64);
 assert(64==vv.size());
 assert(64==vv.end()-vv.begin());
 assert(std::get_if<Vector>(&vv.container()));
 i=0; for (auto c : vv) assert(c==i++);

 va.extend(data+5,data+10);
 assert(10==va.size());
 assert(10==va.end()-va.begin());
 if constexpr (sizeof(T)<4) 
    assert(std::get_if<Array>(&va.container()));
 else
    assert(std::get_if<Vector>(&va.container()));
 i=0; for (auto c : va) assert(c==i++);
 va.extend(data+10,data+64);
 assert(64==va.size());
 assert(64==va.end()-va.begin());
 assert(std::get_if<Vector>(&va.container()));
 i=0; for (auto c : va) assert(c==i++);

 vv.extend(data+64,data+72); 
 assert(72==vv.size());
 assert(72==vv.end()-vv.begin());
 assert(std::get_if<Vector>(&vv.container()));
 i=0; for (auto c : vv) assert(c==i++);

 return 0;

}


int main() {

  go<uint8_t>();
  go<uint16_t>();
  go<int32_t>();

  return 0;
}


