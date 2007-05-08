/** \class LASvector2D
 *  helper class contains vector algebra used by Bruno's alignment algorithm.
 *  matrix format is based on std::vector<std::vector<T>>
 *  LASvector2D[n][m] has n columns, m rows
 *
 *  $Date: Fri Apr 20 14:18:19 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */
 
#ifndef LaserAlignment_LASvector2D_H
#define LaserAlignment_LASvector2D_H

#include "Alignment/LaserAlignment/interface/LASvector.h"
#include <vector>

template<class Element> class LASvector2D : public std::vector<LASvector<Element> >
{
public:
 /// constructor
 LASvector2D(size_t nrows=0, size_t ncolums=0, const Element init=0) : std::vector<LASvector<Element> >(nrows, LASvector<Element>(ncolums,init)) {};
 /// return sum of elements in the columns of a matrix  
 LASvector<Element> sumC();
 /// return sum of elements in the rows of a matrix
 LASvector<Element> sumR();
 /// return sum of all elements in the matrix
 Element sum();
 /// transpose the matrix
 void transpose(const LASvector2D<Element> & source);
 
 /// element wise product between LASvector2D<T> and LASvector<T>
 LASvector2D<Element> operator*(const LASvector<Element> & multiplicator) const;
 /// multiplication of a LASvector2D with a scalar
 LASvector2D<Element> operator*(Element multiplicator) const;
 /// division of a LASvector2D by a scalar
 LASvector2D<Element> operator/(Element factor) const;
};

#endif /* LaserAlignment_LASvector2D_H */
