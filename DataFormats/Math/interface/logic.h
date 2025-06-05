#ifndef DataFormats_Math_logic_h
#define DataFormats_Math_logic_h

namespace reco {
  // this function can be called with any boolean expressions as the parameters
  // this forces the evaluation of both expressions (faster if the expressions are simple)
  // and applying && to two bools avoids branching (jump instruction)
  // whereas applying && to the two original expressions may cause branching
  // this is an alternative to using the bitwise and operator (&), which never short-circuits
  inline bool branchless_and(bool a, bool b) { return a && b; }
}  // namespace reco

#endif
