#ifndef AlignTransformErrorExtended_H
#define AlignTransformErrorExtended_H
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Vector/RotationInterfaces.h"

#include "CondFormats/Alignment/interface/Definitions.h"

/// Class holding error due to an Alignment transformation
/// It contains the raw detector id and the symmetrical error matrix.
/// It is optimized for storage (error matrix is stored as C-array)
class  AlignTransformErrorExtended {
public:
  typedef CLHEP::HepSymMatrix SymMatrix;


  AlignTransformErrorExtended(){ }
  AlignTransformErrorExtended( const SymMatrix & symMatrix,
		       align::ID irawId ) :
    m_rawId(irawId) 
  { 
    for ( unsigned int i=0; i<m_nPars; ++i )
      for ( unsigned int j=0; j<=i; ++j )
        m_Parameters[i*(i+1)/2+j] = symMatrix[i][j];
  }
  
  SymMatrix matrix() const { 
    SymMatrix result(m_nPars);
    for ( unsigned int i=0; i<m_nPars; ++i )
      for ( unsigned int j=0; j<=i; ++j )
        result[i][j] = m_Parameters[i*(i+1)/2+j];
    return result;
  }

  align::ID rawId()  const { return m_rawId; }

  // Implemented so this can be sorted by rawId
  const bool operator < ( const AlignTransformErrorExtended & other ) const
  {
        return ( m_rawId < other.rawId() );
  }

private:

  //FIXME difference here 
  static const unsigned int m_nPars = 6;
  double m_Parameters[m_nPars*(m_nPars+1)/2];
  align::ID m_rawId;



  COND_SERIALIZABLE;
};
#endif //AlignTransformErrorExtended_H
