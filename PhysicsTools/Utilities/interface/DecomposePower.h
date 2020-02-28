#ifndef PhysicsTools_Utilities_DecomposePower_h
#define PhysicsTools_Utilities_DecomposePower_h
#include "PhysicsTools/Utilities/interface/Power.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"

namespace funct {
  template <typename A, typename B>
  struct DecomposePower {
    typedef PowerStruct<A, B> type;
    inline static const A& getBase(const type& _) { return _._1; }
    inline static const B& getExp(const type& _) { return _._2; }
  };

  template <typename A>
  struct DecomposePower<A, Numerical<1> > {
    typedef A type;
    inline static const A& getBase(const type& _) { return _; }
    inline static Numerical<1> getExp(const type& _) { return num<1>(); }
  };

}  // namespace funct

#endif
