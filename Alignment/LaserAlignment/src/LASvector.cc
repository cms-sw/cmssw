/** \file LASvector.cc
 *  helper class contains vector algebra for Bruno's alignment algorithm
 *
 *  $Date: 2007/05/08 08:03:24 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */
 
#include "Alignment/LaserAlignment/interface/LASvector.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <math.h>
#include <ostream>

template<class Element>
LASvector<Element> LASvector<Element>::operator+(const LASvector<Element> & source) const
{
  // check if size matches, else throw exception
  if (this->size() != source.size())
    throw cms::Exception("SizeMisMatch","Size of vectors do not match!") << this->size() 
      << " does not match the size of the multiplicator " << source.size() << std::endl;

  LASvector<Element> result(*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    result[i] += source[i];
  }  
  return result;
}

template<class Element>
LASvector<Element> LASvector<Element>::operator+(Element source) const
{
  LASvector<Element> result(*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    result[i] += source;
  }  
  return result;
}

template<class Element>
LASvector<Element> LASvector<Element>::operator-(const LASvector<Element> & source) const
{
  // check if size matches, else throw exception
  if (this->size() != source.size())
    throw cms::Exception("SizeMisMatch","Size of vectors do not match!") << this->size() 
      << " does not match the size of the multiplicator " << source.size() << std::endl;

  LASvector<Element> result(*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    result[i] -= source[i];
  }  
  return result;
}

template<class Element>
LASvector<Element> LASvector<Element>::operator-(Element source) const
{
  LASvector<Element> result(*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    result[i] -= source;
  }  
  return result;
}

template<class Element>
LASvector<Element> LASvector<Element>::operator*(const LASvector<Element> &source) const
{
  // check if size matches, else throw exception
  if (this->size() != source.size())
    throw cms::Exception("SizeMisMatch","Size of vectors do not match!") << this->size() 
      << " does not match the size of the multiplicator " << source.size() << std::endl;
	
  LASvector<Element> result(*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    result[i]*=source[i];
  }
  return result;
}

template<class Element>
LASvector<Element> LASvector<Element>::operator*(Element multiplicator) const
{
  LASvector<Element> result(*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
    result[i] *= multiplicator;
  }
  return result;
}

template<class Element>
LASvector<Element> LASvector<Element>::operator/(Element factor) const
{
  if (factor == 0)
    throw cms::Exception("DivisionByZero","Division by zero is not allowed!!!");
    
  LASvector<Element> result = (*this);
  for( unsigned int i = 0; i < result.size(); ++i )
  {
      result[i] /= factor;
  }
  return result;
}

template<class Element>
void LASvector<Element>::operator-=(Element subtractor)
{
  for( unsigned int i = 0; i < (*this).size(); ++i )
  {
    (*this)[i] -= subtractor;
  }
}

template<class Element>
void LASvector<Element>::sine(const LASvector<Element> &source)
{
  LASvector<Element> result;
  for( unsigned int i = 0; i < source.size(); ++i )
  {
    (*this)[i] = sin(source[i]);
  }
}

template<class Element>
void LASvector<Element>::cosine(const LASvector<Element> &source)
{
  for( unsigned int i = 0; i < source.size(); ++i )
  {
    (*this)[i] = cos(source[i]);
  }
}

template<class Element>
Element LASvector<Element>::sum()
{
  Element result = 0;
  for( unsigned int i = 0; i < (*this).size(); ++i )
  {
    result += (*this)[i];
  }
  return result;
}

template LASvector<double>::LASvector(size_t size=0, const double init=0);
template LASvector<double> LASvector<double>::operator+(const LASvector<double> & source) const;
template LASvector<double> LASvector<double>::operator+(double source) const;
template LASvector<double> LASvector<double>::operator-(const LASvector<double> & source) const;
template LASvector<double> LASvector<double>::operator-(double source) const;
template LASvector<double> LASvector<double>::operator*(const LASvector<double> &source) const;
template LASvector<double> LASvector<double>::operator*(double mulitplicator) const;
template LASvector<double> LASvector<double>::operator/(double factor) const;
template void LASvector<double>::operator-=(double subtractor);
template void LASvector<double>::sine(const LASvector<double> & source);
template void LASvector<double>::cosine(const LASvector<double> & source);
template double LASvector<double>::sum();

// template LASvector<LASvector<double> >::LASvector(size_t size=0, const LASvector<double>(size_t size=0, const double init=0));
