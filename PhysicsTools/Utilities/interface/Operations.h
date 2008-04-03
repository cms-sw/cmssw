#ifndef PhysicsTools_Utilities_Operations_h
#define PhysicsTools_Utilities_Operations_h
#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/Difference.h"
#include "PhysicsTools/Utilities/interface/Ratio.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Power.h"
#include "PhysicsTools/Utilities/interface/Minus.h"

namespace funct {

  template<typename A, typename B>
  inline typename Sum<A, B>::type operator+(const A& a, const B& b) {
    return Sum<A, B>::combine(a, b);
  }

  template<typename A, typename B>
  inline typename Difference<A, B>::type operator-(const A& a, const B& b) {
    return Difference<A, B>::combine(a, b);
  }
  
  template<typename A, typename B>
  inline typename Product<A, B>::type operator*(const A& a, const B& b) {
    return Product<A, B>::combine(a, b);
  }

  template<typename A, typename B>
  inline typename Ratio<A, B>::type operator/(const A& a, const B& b) {
    return Ratio<A, B>::combine(a, b);
  }
  
  template<typename A>
  inline typename Minus<A>::type operator-(const A& a) {
    return Minus<A>::operate(a);
  }

  template<typename A, typename B>
  inline typename Power<A, B>::type operator^(const A& a, const B& b) {
    return Power<A, B>::combine(a, b);
  }
  
}

/*
#include "PhysicsTools/Utilities/interface/Number.h"
#include "PhysicsTools/Utilities/interface/Constant.h"

namespace funct {

  template<typename A>
  typename Difference<A, Constant>::type 
    inline operator-(const A& a, const Parameter& b) {
    return Difference<A, Constant>::combine(a, Constant(b));
  }

  template<typename B>
  typename Difference<Number, B>::type 
    inline operator-(double a, const B& b) {
    return Difference<Number, B>::combine(Number(a), b);
  }
  
  Difference<Number, Constant>::type
  inline operator-(double a, const Parameter& b) {
    return Difference<Number, 
      Constant>::combine(Number(a), Constant(b));
  }
  
  template<typename B>
  inline typename Product<Constant, B>::type operator*(const Parameter& a, const B& b) {
    return Product<Constant, B>::combine(Constant(a), b);
  }

}

*/ 

#endif
