#ifndef DataFormatsMathapprox_asin_h
#define DataFormatsMathapprox_asin_h

#include<cmath>
template<int DEGREE>
constexpr float approx_asin_P(float z);


   // degree =  3   => absolute accuracy is  8 bits
template<> constexpr float approx_asin_P< 3 >(float z){
 return  1.f + z * 0.2133004963397979736328125f;
}
   // degree =  5   => absolute accuracy is  11 bits
template<> constexpr float approx_asin_P< 5 >(float z){
 return 1.f + z * (0.154711246490478515625f + z * 0.1322681009769439697265625f);
}
   // degree =  7   => absolute accuracy is  15 bits
template<> constexpr float approx_asin_P< 7 >(float z){
 return  1.f + z * (0.169519245624542236328125f + z * (4.9031913280487060546875e-2f + z * 0.10941398143768310546875));
}
   // degree =  9   => absolute accuracy is  18 bits
template<> constexpr float approx_asin_P< 9 >(float z){
 return  1.f + z * (0.166020572185516357421875f + z * (8.44048559665679931640625e-2f + z * (1.11602735705673694610595703125e-3f + z * 0.103476583957672119140625f)));
}
   // degree =  11   => absolute accuracy is  21 bits
template<> constexpr float approx_asin_P< 11 >(float z){
 return  1.f + z * (0.1668075025081634521484375f + z * (7.20207393169403076171875e-2f + z * (6.607978045940399169921875e-2f + z * ((-3.6048568785190582275390625e-2f) + z * 0.10574872791767120361328125f))));
}

/*
   // degree =  3   => absolute accuracy is  8 bits
template<> constexpr float approx_asin_P< 3 >(float z){
 return  1.f + z * 0.2114248573780059814453125f;
}
   // degree =  5   => absolute accuracy is  12 bits
template<> constexpr float approx_asin_P< 5 >(float z){
 return  1.f + z * (0.1556626856327056884765625f + z * 0.1295671761035919189453125f);
}
   // degree =  7   => absolute accuracy is  15 bits
template<> constexpr float approx_asin_P< 7 >(float z){
 return  1.f + z * (0.1691854894161224365234375f + z * (5.1305986940860748291015625e-2f + z * 0.1058919131755828857421875f));
}
   // degree =  9   => absolute accuracy is  18 bits
template<> constexpr float approx_asin_P< 9 >(float z){
 return  1.f + z * (0.166119158267974853515625f + z * (8.322779834270477294921875e-2f + z * (5.28236292302608489990234375e-3f + z * 9.89462435245513916015625e-2f)));
}
   // degree =  11   => absolute accuracy is  21 bits
template<> constexpr float approx_asin_P< 11 >(float z){
 return  1.f + z * (0.1667812168598175048828125f + z * (7.249967753887176513671875e-2f + z * (6.321799755096435546875e-2f + z * ((-2.913488447666168212890625e-2f) + z * 9.9913299083709716796875e-2f))));
}
*/



// valid for |x|<0.71
template<int DEGREE>
constexpr float unsafe_asin07(float x) {
  auto z=x*x;
  return x*approx_asin_P<DEGREE>(z);
}

template<int DEGREE>
constexpr float unsafe_acos07(float x) {
  constexpr float pihalf = M_PI/2; 
  return  pihalf - unsafe_asin07<DEGREE>(x);
}


// for |x|> 0.71 use slower
template<int DEGREE>
constexpr float unsafe_acos71(float x) {
  constexpr float pi = M_PI;
  auto z=1.f-x*x;
  z= std::sqrt(z)*approx_asin_P<DEGREE>(z);
  return x>0 ? z : pi-z;
}

template<int DEGREE>
constexpr float unsafe_asin71(float x) {
  constexpr float pihalf = M_PI/2;
  return  pihalf - unsafe_acos71<DEGREE>(x);
}

template<int DEGREE>
constexpr float unsafe_asin(float x) {
  return (std::abs(x)<0.71f) ? unsafe_asin07<DEGREE>(x) : unsafe_asin71<DEGREE>(x);
}

template<int DEGREE>
constexpr float unsafe_acos(float x) {
  return (std::abs(x)<0.71f) ? unsafe_acos07<DEGREE>(x) :	unsafe_acos71<DEGREE>(x);
}


#endif
