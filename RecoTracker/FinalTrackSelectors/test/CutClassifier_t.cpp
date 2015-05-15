#include<iostream>
#include<cassert>


namespace {
  constexpr float mvaVal[3] = {-.5,.5,1.};
  
  template<typename T,typename Comp>
  inline float cut(T val, const T * cuts, Comp comp) {
    for (int i=2; i>=0; --i) 
      if ( comp(val,cuts[i]) ) return mvaVal[i];
    return -1.f; 
  }

}


int main() {

  float maxChi2[3] ={9999.,25.,16.};
  int min3DLayers[3] = {1,2,3}; 
  int maxLostLayers[3] = {99,3,3};


  float ret = -1.f;
  ret = cut(2.f,maxChi2,std::less_equal<float>());
  assert(ret==mvaVal[2]);
  ret = cut(28.f,maxChi2,std::less_equal<float>());
  assert(ret==mvaVal[0]);
  
  ret = cut(0,min3DLayers,std::greater_equal<int>());
  assert(ret==-1);
  ret = cut(1,min3DLayers,std::greater_equal<int>());
  assert(ret==mvaVal[0]);
  ret = cut(2,min3DLayers,std::greater_equal<int>());
  assert(ret==mvaVal[1]);
  ret = cut(3,min3DLayers,std::greater_equal<int>());
  assert(ret==mvaVal[2]);
  ret = cut(5,min3DLayers,std::greater_equal<int>());
  assert(ret==mvaVal[2]);
  
  ret = cut(1,maxLostLayers,std::less_equal<int>());
  assert(ret==mvaVal[2]);
      
  return 0;

}
