#ifndef IncompleteGammaComplement_H
#define IncompleteGammaComplement_H

/** \class IncompleteGammaComplement
 *  Computes the complement Q(a,x)=1-P(a,x) of the incomplete gamma function. 
 *  Computes its natural logarithm, useful for comparing very small values. 
 *  source: Numerical Recipes
 */
class IncompleteGammaComplement {

public:

  static float value( float a, float x );

  static float ln( float a, float x );

};

#endif
