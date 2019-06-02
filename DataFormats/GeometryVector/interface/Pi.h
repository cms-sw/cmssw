#ifndef GeometryVector_Geom_Pi_h
#define GeometryVector_Geom_Pi_h

/** \file 
 *  Just pi (the ratio of circle length to diameter).
 *  Since the C++ standard seems not to care about PI and the
 *  M_PI macro definition in <cmath> is not even in the standard
 *  a definition is provided here.
 *  <li>
 *  The goal of this implementation is to be as efficient as a macro,
 *  and transparent to use. Excluding macros, there are four ways
 *  to implement a constant for shared use:
 *    1) As a global constant (possibly within a namespace), that
 *       is initialized by the linker/loader and used as a memory reference
 *    2) As a file scope constant (i.e. in the unnamed namespace)
 *       with one copy per file
 *    3) As an in-line function that returns the value
 *    4) As a class with an in-line method that returns the value
 *  The first way is probably the most efficient but it requires
 *  an additional libraby to link, something you should not need
 *  if you just want to use pi.
 *  The fourth way is only justified if you need more than one method
 *  for the same constant (e.g. return it in different units)
 *  The difference between the second and the third is essentially in the
 *  parantheses of the function call, which are not needed in the
 *  second method.
 */

namespace Geom {

  inline constexpr double pi() { return 3.141592653589793238; }
  inline constexpr double twoPi() { return 2. * 3.141592653589793238; }
  inline constexpr double halfPi() { return 0.5 * 3.141592653589793238; }

  inline constexpr float fpi() { return 3.141592653589793238f; }
  inline constexpr float ftwoPi() { return 2.f * 3.141592653589793238f; }
  inline constexpr float fhalfPi() { return 0.5f * 3.141592653589793238f; }

}  // namespace Geom

#endif
