#include<iostream>
#include<cassert>
#include<tuple>
#include<array>
#include<utility>

namespace {

  struct A {
    float chi2;
    int   n3d;
    int   nll;
  };


  constexpr float mvaVal[3] = {-.5,.5,1.};
  
  template<typename T,typename Comp>
  inline float cut(T val, const T * cuts, Comp comp) {
    for (int i=2; i>=0; --i) 
      if ( comp(val,cuts[i]) ) return mvaVal[i];
    return -1.f; 
  }



  template<typename CutT>
  inline float cut(A const & a, CutT const & cuts) {
    for (int i=2; i>=0; --i) 
      if ( std::get<1>(cuts)(std::get<0>(cuts)(a),std::get<2>(cuts)[i]) ) return mvaVal[i];
    return -1.f; 
  }


  template<class F, class...Ts, std::size_t...Is>
  void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func, std::index_sequence<Is...>){
    using expander = int[];
    (void)expander { 0, ((void)func(std::get<Is>(tuple)), 0)... };
  }
  
  template<class F, class...Ts>
  void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func){
    for_each_in_tuple(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
  }

    



  float applyCuts(A const & a) {
    constexpr auto allCuts = std::make_tuple(
					     std::make_tuple(
							     [](A const & a) { return a.chi2;},
							     std::less_equal<>(),
							     std::array<float,3>{9999.,25.,16.}
							     ),
					     std::make_tuple(
							     [](A const & a) { return a.n3d;},
							     std::greater_equal<>(),
							     std::array<int,3>{1,2,3}
							     )
					     );
    float aret = 1;
    for_each_in_tuple(allCuts,[&](auto const & c) { if (aret>-1.f) aret = std::min(aret,cut(a,c));});
    return aret;
  }

}

int main() {

  A a{18.,1,1};

  auto cut1 = std::make_tuple(
			      [](A const & a) { return a.chi2;},
			      std::less_equal<>(),
			      std::array<float,3>{9999.,25.,16.}
			      );
  std::cout << std::get<1>(cut1)(std::get<0>(cut1)(a),std::get<2>(cut1)[0]) << std::endl;
  std::cout << std::get<1>(cut1)(std::get<0>(cut1)(a),std::get<2>(cut1)[2]) << std::endl;

  
  std::cout << applyCuts(a) << std::endl;
  
  
  float maxChi2[3] ={9999.,25.,16.};
  int min3DLayers[3] = {1,2,3}; 
  int maxLostLayers[3] = {99,3,3};


  float ret = -1.f;
  ret = cut(2.f,maxChi2,std::less_equal<>());
  assert(ret==mvaVal[2]);
  ret = cut(28.f,maxChi2,std::less_equal<>());
  assert(ret==mvaVal[0]);
  
  ret = cut(0,min3DLayers,std::greater_equal<>());
  assert(ret==-1);
  ret = cut(1,min3DLayers,std::greater_equal<>());
  assert(ret==mvaVal[0]);
  ret = cut(2,min3DLayers,std::greater_equal<>());
  assert(ret==mvaVal[1]);
  ret = cut(3,min3DLayers,std::greater_equal<>());
  assert(ret==mvaVal[2]);
  ret = cut(5,min3DLayers,std::greater_equal<>());
  assert(ret==mvaVal[2]);
  
  ret = cut(1,maxLostLayers,std::less_equal<>());
  assert(ret==mvaVal[2]);
      
  return 0;

}
