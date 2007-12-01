#ifndef AlignTransformError_H
#define AlignTransformError_H
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Vector/RotationInterfaces.h"
#include <boost/cstdint.hpp>


/// Class holding error due to an Alignment transformation
/// It contains the raw detector id and the symmetrical error matrix.
/// It is optimized for storage (error matrix is stored as C-array)
class  AlignTransformError {
public:
  typedef CLHEP::HepSymMatrix SymMatrix;


  AlignTransformError(){ }
  AlignTransformError( const SymMatrix & symMatrix,
					   const uint32_t & irawId ) :
    m_rawId(irawId) 
  { 
    for ( unsigned int i=0; i<m_nPars; ++i )
      for ( unsigned int j=0; j<=i; ++j )
        m_Parameters[i*(i+1)/2+j] = symMatrix[i][j];
  }
  
  const SymMatrix matrix() const { 
    SymMatrix result(m_nPars);
    for ( unsigned int i=0; i<m_nPars; ++i )
      for ( unsigned int j=0; j<=i; ++j )
        result[i][j] = m_Parameters[i*(i+1)/2+j];
    return result;
  }

  const uint32_t&  rawId()  const { return m_rawId; }

private:

  static const unsigned int m_nPars = 3;
  double m_Parameters[m_nPars*(m_nPars+1)/2];
  uint32_t m_rawId;


};
#endif //AlignTransformError_H
