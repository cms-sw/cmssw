#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

#include<vector>

#include<iostream>

// this is a test,
// using namespace mathSSE;

void addScaleddiff(Basic3DVectorF&res, float s,  Basic3DVectorF const & a, Basic3DVectorF const & b) {
  res += s*(a-b);
} 

void addScaleddiff2(Basic3DVectorF&res, float s,  Basic3DVectorF const & a, Basic3DVectorF const & b) {
  res = res + s*(a-b);
}

void addScaleddiff(Basic3DVectorD&res, float s,  Basic3DVectorD const & a, Basic3DVectorD const & b) {
  res += s*(a-b);
} 

void multiSum(Basic3DVectorF&res, float s,  Basic3DVectorF const & a, Basic3DVectorF const & b) {
  res = s*(a-b) + s*(a+b);
} 

void multiSum(Basic3DVectorD&res, float s,  Basic3DVectorD const & a, Basic3DVectorD const & b) {
  res = s*(a-b) + s*(a+b);
} 

void multiSum(Basic3DVectorLD&res, float s,  Basic3DVectorLD const & a, Basic3DVectorLD const & b) {
  res = s*(a-b) + s*(a+b);
} 



template<typename T, typename U>
typename PreciseFloatType<T,U>::Type dotV(  Basic3DVector<T> const & a,  Basic3DVector<U> const & b) {
  return a*b;
}

template<typename T>
T norm(Basic3DVector<T> const & a) {
  return std::sqrt(a*a);
}

template<typename T>
T normV(Basic3DVector<T> const & a) {
  return a.mag();
}

template<typename T, typename U>
typename PreciseFloatType<T,U>::Type dotV(  Basic2DVector<T> const & a,  Basic2DVector<U> const & b) {
  return a*b;
}

template<typename T>
T norm(Basic2DVector<T> const & a) {
  return std::sqrt(a*a);
}

template<typename T>
T normV(Basic2DVector<T> const & a) {
  return a.mag();
}


long aligned(void * p) {
  return long(p)&0xf;

}

volatile int * vi=0;

template<typename T> 
void verifyAlign() {
  int sum=0;
  int nota=0;
  for (int i=0; i!=100; ++i) {
    auto p= new int(3);
    sum +=(*p);
    auto t = new T;
    sum +=(*t)[0];
    if (aligned(t)!=0) ++nota;
    delete p;
    delete t;
    t = new T;
    sum +=(*t)[0];
    if (aligned(t)!=0) ++nota;
    delete t;
    vi = new int[3];
    auto vt = new T[3];
    if (aligned(vt)!=0) ++nota;
    sum +=vt[1][1];
    delete [] vi;
    delete [] vt;
  }

  std::cout << "sum " << sum << std::endl;
  std::cout << "not aligned " << nota << std::endl;

}

int main() {
#if defined(__GNUC__)
  std::cout << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
#endif
#ifdef USE_SSEVECT
  std::cout << "sse vector enabled in cmssw" << std::endl;
#endif
#ifdef USE_EXTVECT
  std::cout << "extended vector notation enabled in cmssw" << std::endl;
#endif


  std::cout << "biggest alignment " << __BIGGEST_ALIGNMENT__ << std::endl;




  std::cout << sizeof(Basic2DVectorF) << std::endl;
  std::cout << sizeof(Basic2DVectorD) << std::endl;
  std::cout << sizeof(Basic3DVectorF) << std::endl;
  std::cout << sizeof(Basic3DVectorD) << std::endl;
  std::cout << sizeof(Basic3DVectorLD) << std::endl;

  verifyAlign<Basic3DVectorF>();
  verifyAlign<Basic3DVectorD>();


  Basic3DVectorF  x(2.0f,4.0f,5.0f);
  Basic3DVectorF  y(-3.0f,2.0f,-5.0f);
  Basic3DVectorD  xd(2.0,4.0,5.0);
  Basic3DVectorD  yd = y;

  Basic3DVectorLD  xld(2.0,4.0,5.0);
  Basic3DVectorLD  yld = y;


  Basic2DVectorF  x2(2.0f,4.0f);
  Basic2DVectorF  y2 = y.xy();
  Basic2DVectorD  xd2(2.0,4.0);
  Basic2DVectorD  yd2 = yd.xy();

  {
    std::cout << dotV(x,y) << std::endl; 
    std::cout << normV(x) << std::endl; 
    std::cout << norm(x) << std::endl; 
    // std::cout << std::min(x.mathVector(),y.mathVector()) << std::endl;
    // std::cout << std::max(x.mathVector(),y.mathVector()) << std::endl;

    std::cout << dotV(x,yd) << std::endl; 
    std::cout << dotV(xd,y) << std::endl; 
    std::cout << dotV(xd,yd) << std::endl; 
    std::cout << normV(xd) << std::endl; 
    std::cout << norm(xd) << std::endl; 
    std::cout << dotV(xld,yld) << std::endl; 
    std::cout << normV(xld) << std::endl; 
    std::cout << norm(xld) << std::endl; 
    
    
    Basic3DVectorF  z = x.cross(y);
    std::cout << z << std::endl;
    std::cout << -z << std::endl;
    Basic3DVectorD  zd = x.cross(yd);
    std::cout << zd << std::endl;
    std::cout << -zd << std::endl;
    std::cout << xd.cross(y)<< std::endl;
    std::cout << xd.cross(yd)<< std::endl;

    Basic3DVectorLD  zld = x.cross(yld);
    std::cout << zld << std::endl;
    std::cout << -zld << std::endl;
    std::cout << xld.cross(y)<< std::endl;
    std::cout << xld.cross(yld)<< std::endl;

    std::cout << z.eta() << " " << (-z).eta() << std::endl;
    std::cout << zd.eta()  << " " << (-zd).eta() << std::endl;
    std::cout << zld.eta()  << " " << (-zld).eta() << std::endl;
    
#if defined( __GXX_EXPERIMENTAL_CXX0X__)
    auto s = x+xd - 3.1*z;
    std::cout << s << std::endl;
    auto s2 = x+xld - 3.1*zd;
    std::cout << s2 << std::endl;

#endif
  }

 {
    std::cout << dotV(x2,y2) << std::endl; 
    std::cout << normV(x2) << std::endl; 
    std::cout << norm(x2) << std::endl; 
    // std::cout << std::min(x2.mathVector(),y2.mathVector()) << std::endl;
    // std::cout << std::max(x2.mathVector(),y2.mathVector()) << std::endl;

    std::cout << dotV(x2,yd2) << std::endl; 
    std::cout << dotV(xd2,y2) << std::endl; 
    std::cout << dotV(xd2,yd2) << std::endl; 
    std::cout << normV(xd2) << std::endl; 
    std::cout << norm(xd2) << std::endl; 
    
    
    Basic2DVectorF  z2(x2); z2-=y2;
    std::cout << z2 << std::endl;
    std::cout << -z2 << std::endl;
    Basic2DVectorD zd2 = x2-yd2;
    std::cout << zd2 << std::endl;
    std::cout << -zd2 << std::endl;
    std::cout << x2.cross(y2) << std::endl;
    std::cout << x2.cross(yd2) << std::endl;
    std::cout << xd2.cross(y2)<< std::endl;
    std::cout << xd2.cross(yd2)<< std::endl;
    
    auto s2 = x2+xd2 - 3.1*z2;
    std::cout << s2 << std::endl;
  }



  {
    std::cout << "f" << std::endl;
    Basic3DVectorF  vx(2.0f,4.0f,5.0f);
    Basic3DVectorF  vy(-3.0f,2.0f,-5.0f);
    vx+=vy;
    std::cout << vx << std::endl;
    
    Basic3DVectorF vz(1.f,1.f,1.f);
    addScaleddiff(vz,0.1f,vx,vy);
    std::cout << vz << std::endl;
  }

 {
    std::cout << "d" << std::endl;
    Basic3DVectorD  vx(2.0,4.0,5.0);
    Basic3DVectorD  vy(-3.0,2.0,-5.0);
    vx+=vy;
    std::cout << vx << std::endl;
    
    Basic3DVectorD vz(1.,1.,1);
    addScaleddiff(vz,0.1,vx,vy);
    std::cout << vz << std::endl;
 }

 std::cout << "std::vector" << std::endl;
 std::vector<Basic3DVectorF> vec1; vec1.reserve(50);
 std::vector<float> vecf(21);
 std::vector<Basic3DVectorF> vec2(51);
 std::vector<Basic3DVectorF> vec3; vec3.reserve(23456);
}
