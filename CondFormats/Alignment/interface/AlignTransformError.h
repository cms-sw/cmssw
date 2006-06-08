#ifndef AlignTransformError_H
#define AlignTransformError_H
#include "CLHEP/Matrix/SymMatrix.h"
#include <boost/cstdint.hpp>


/// Class holding error due to an Alignment transformation
/// It contains the raw detector id and the symmetrical error matrix.
/// It is optimized for storage.
class  AlignTransformError {
public:
  typedef CLHEP::HepSymMatrix SymMatrix;


  AlignTransformError(){ }
  AlignTransformError( const SymMatrix & symMatrix,
					   const uint32_t & irawId ) :
    m_symMatrix(symMatrix), m_rawId(irawId) { }
  
  const SymMatrix& matrix() const { return m_symMatrix; }
  const uint32_t&  rawId()  const { return m_rawId; }

private:

  SymMatrix m_symMatrix;
  uint32_t m_rawId;


};
#endif //AlignTransformError_H
