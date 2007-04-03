#ifndef CondFormats_Alignment_SurveyError_H
#define CondFormats_Alignment_SurveyError_H

/** \class SurveyError
 *
 *  Class to hold DB object for survey errors.
 *
 *  DB object contains the following:
 *    an unsigned 8-bit integer for the structure type
 *    an unsigned 32-bit integer for the detector's raw id
 *    an array of 21 floats for the error matrix of 6 alignment parameters
 *  The lower triangular of the error matrix is stored.
 *
 *  $Date: 2007/03/22 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include <boost/cstdint.hpp>

#include "CondFormats/Alignment/interface/Definitions.h"

class SurveyError
{
  typedef align::ErrorMatrix ErrorMatrix;
  typedef ErrorMatrix::value_type Scalar;

  public:

  inline SurveyError(
		     uint8_t structureType = 0, // default unknown
		     uint32_t rawId = 0,        // default unknown
		     const ErrorMatrix& = ErrorMatrix() // default 0
		     );

  inline const uint8_t& structureType() const;

  inline const uint32_t&  rawId() const;
  
  inline ErrorMatrix matrix() const;

private:

  static const unsigned int nPar_ = ErrorMatrix::kRows;
  static const unsigned int size_ = nPar_ * (nPar_ + 1) / 2;

  uint8_t  m_structureType;
  uint32_t m_rawId;

  Scalar m_errors[size_];
};

SurveyError::SurveyError(uint8_t structureType,
			 uint32_t rawId,
			 const ErrorMatrix& cov):
  m_structureType(structureType),
  m_rawId(rawId)
{
  const Scalar* data = cov.Array(); // lower triangular of cov

  for (unsigned int i = 0; i < size_; ++i) m_errors[i] = data[i];
}

const uint8_t& SurveyError::structureType() const
{
  return m_structureType;
}

const uint32_t& SurveyError::rawId() const
{
  return m_rawId;
}

SurveyError::ErrorMatrix SurveyError::matrix() const
{
  return ErrorMatrix(m_errors, m_errors + size_);
}

#endif
