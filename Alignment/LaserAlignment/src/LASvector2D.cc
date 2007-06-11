/** \file LASvector2D.cc
 *  matrix algebra for Bruno's alignment algorithm
 *
 *  $Date: 2007/05/08 08:03:24 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/LaserAlignment/interface/LASvector2D.h"

template<class Element>
LASvector2D<Element> LASvector2D<Element>::operator*(const LASvector<Element> & multiplicator) const
{
  if ((*this)[0].size() != multiplicator.size())
    throw cms::Exception("SizeMisMatch","Size of matrix and multiplicator (vector) do not match!") << this[0].size() 
      << " does not match the size of the multiplicator " << multiplicator.size() << std::endl;
  LASvector2D<Element> result = (*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    for( unsigned int j = 0; j < result[0].size(); ++j )
    {
      result[i][j] *= multiplicator[j];
    }
  }
  return result;
}

template<class Element>
LASvector2D<Element> LASvector2D<Element>::operator*(Element multiplicator) const 
{
  LASvector2D<Element> result = (*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    for( unsigned int j = 0; j < result[0].size(); ++j )
    {
      result[i][j] *= multiplicator;
    }
  }
  return result;
}

template<class Element>
LASvector2D<Element> LASvector2D<Element>::operator/(Element factor) const
{
  if (factor == 0)
    throw cms::Exception("DivisionByZero","Division by zero is not allowed!!!");
    
  LASvector2D<Element> result = (*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    for( unsigned int j = 0; j < result[0].size(); ++j )
    {
      result[i][j] /= factor;
    }
  }
  return result;
}


template<class Element>
LASvector<Element> LASvector2D<Element>::sumC()
{
  LASvector<Element> result((*this).size());
  for( unsigned int i = 0; i < (*this).size(); ++i )
  {
    for( unsigned int j = 0; j < (*this)[i].size(); ++j )
    {
      result[i] += (*this)[i][j];
    }
  }
  return result;
}

template<class Element>
LASvector<Element> LASvector2D<Element>::sumR()
{
  LASvector<Element> result((*this)[0].size());
  for( unsigned int i = 0; i < (*this)[0].size(); ++i )
  {
    for( unsigned int j = 0; j < (*this).size(); ++j )
    {
      result[i] += (*this)[i][j];
    }
  }
  return result;
}

template<class Element>
Element LASvector2D<Element>::sum()
{
  Element result = 0;
  for( unsigned int i = 0; i < (*this).size(); ++i )
  {
    for( unsigned int j = 0; j < (*this)[i].size(); ++j )
    {
      result += (*this)[i][j];
    }
  }
  return result;
}

template<class Element>
void LASvector2D<Element>::transpose(const LASvector2D<Element> & source)
{
  for( unsigned int i = 0; i < source.size(); ++i )
  {
    for( unsigned int j = 0; j < source[i].size(); ++j )
    {
      (*this)[j][i] = source[i][j];
    }
  }
}

template LASvector2D<double>::LASvector2D(size_t nrows=0, size_t ncolumns=0, const double init=0);
template LASvector<double> LASvector2D<double>::sumC();
template LASvector<double> LASvector2D<double>::sumR();
template double LASvector2D<double>::sum();
template void LASvector2D<double>::transpose(const LASvector2D<double> & source);
template LASvector2D<double> LASvector2D<double>::operator*(const LASvector<double> & multiplicator) const;
template LASvector2D<double> LASvector2D<double>::operator*(double multiplicator) const;
template LASvector2D<double> LASvector2D<double>::operator/(double factor) const;
