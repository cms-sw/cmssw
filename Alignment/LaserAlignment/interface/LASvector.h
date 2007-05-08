/** \class LASvector
 *  helper class contains vector algebra used by Bruno's alignment algorithm. 
 *  vector format is based on std::vector<double>
 *
 *  $Date: Fri Apr 20 10:18:03 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */
 
#ifndef LaserAlignment_LASvector_H
#define LaserAlignment_LASvector_H

#include <vector>

template<class Element> class LASvector : public std::vector<Element> 
{
public:
  /// constructor
  LASvector<Element> (size_t size=0, const Element init=0) : std::vector<Element>(size,init) {};
  /// define element wise sum of two LASvectors
  LASvector<Element> operator+(const LASvector<Element> & source) const;
  /// define sum between LASvector and a scalar
  LASvector<Element> operator+(Element source) const;
  /// define element wise subtraction of two LASvectors
  LASvector<Element> operator-(const LASvector<Element> & source) const;
  /// define difference between LASvector and a scalar
  LASvector<Element> operator-(Element source) const;
  /// define element wise multiplication of two LASvectors
  LASvector<Element> operator*(const LASvector<Element> &source) const;
  /// define multiplication of a LASvector with a scalar
  LASvector<Element> operator*(Element mulitplicator) const;
  /// define division of a LASvector by a scalar
  LASvector<Element> operator/(Element factor) const;
  /// define operator -= of a LASvector;
  void operator-=(Element subtractor);
  /// define element wise sin function for a LASvector, i.e. returned vector contains \f$sin(x_i)\f$ corresponding to \f$x_i\f$ of source vector
  void sine(const LASvector<Element> & source);
  /// define element wise cos function for a LASvector, i.e. returned vector contains \f$cos(x_i)\f$ corresponding to \f$x_i\f$ of source vector
  void cosine(const LASvector<Element> & source);
  /// return the sum of all entries in the vector
  Element sum();

};
#endif /* LaserAlignment_LASvector_H */
