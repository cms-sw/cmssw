#ifndef PhysicsTools_Utilities_DecomposeProduct_h
#define PhysicsTools_Utilities_DecomposeProduct_h
#include "PhysicsTools/Utilities/interface/Product.h"

namespace funct {

  template<typename A, typename B>
  struct DecomposeProduct { };

  template<typename A>
  struct DecomposeProduct<A, A> { 
    inline static const A& get(const A & a) { return a; }
  };

  template<typename A, typename B>
  struct DecomposeProduct<ProductStruct<A, B>, A> { 
    inline static const A& get(const ProductStruct<A, B> & _ ) { return _._1; }
  };

  template<typename A, typename B>
  struct DecomposeProduct<ProductStruct<A, B>, B> { 
    inline static const B& get(const ProductStruct<A, B> & _ ) { return _._2; }
  };

}

#endif
