#ifndef AlignTransform_H
#define AlignTransform_H
#include "CLHEP/Vector/EulerAngles.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Matrix/SymMatrix.h"   // TEMPORARILY also include alignment error
#include <boost/cstdint.hpp>


/// Class holding data for an Alignment transformation
/// It contains the raw detector id, its global position and global orientation.
/// The alignment error is also stored as a temporary workaround.
/// It is optimized for storage.
class  AlignTransform 
{
public:
  typedef CLHEP::HepEulerAngles EulerAngles;
  typedef CLHEP::Hep3Vector     ThreeVector;
  typedef CLHEP::HepRotation    Rotation;
  typedef CLHEP::HepSymMatrix   SymMatrix;
  typedef ThreeVector Translation;

  /// Default constructor
  AlignTransform(){}

  /// Constructor from Euler angles and error matrix
  AlignTransform( const Translation & itranslation, 
				  const EulerAngles & ieulerAngles,
				  const SymMatrix   & ierrorMatrix,
                  const uint32_t & irawId ) :
	m_translation(itranslation),
    m_eulerAngles(ieulerAngles),
	m_errorMatrix(ierrorMatrix),
    m_rawId(irawId) {}
  
  /// Constructor from Rotation and error matrix
  AlignTransform( const Translation & itranslation, 
				  const Rotation    & irotation,
				  const SymMatrix   & ierrorMatrix,
                  const uint32_t & irawId ) :
	m_translation(itranslation),
	m_eulerAngles(irotation.eulerAngles()),
	m_errorMatrix(ierrorMatrix),
    m_rawId(irawId) {}

  const Translation & translation() const { return m_translation; }
  const EulerAngles & eulerAngles() const { return m_eulerAngles; }
  const SymMatrix   & errorMatrix() const { return m_errorMatrix; }
  const uint32_t & rawId() const { return m_rawId; }
  Rotation rotation() const { return Rotation(m_eulerAngles); }

private:

  Translation m_translation;
  EulerAngles m_eulerAngles;
  SymMatrix   m_errorMatrix;
  uint32_t    m_rawId;


};
#endif //AlignTransform_H
