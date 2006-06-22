#ifndef AlignTransform_H
#define AlignTransform_H
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/ThreeVector.h"
#include <boost/cstdint.hpp>


/// Class holding data for an Alignment transformation
/// It contains the raw detector id, its global position and global orientation (3 angles),
/// and the corresponding errors. It is optimized for storage.
class  AlignTransform 
{
public:
  typedef CLHEP::Hep3Vector     ThreeVector;
  typedef CLHEP::HepRotation    Rotation;
  typedef ThreeVector Translation;
  typedef ThreeVector Angles;

  /// Default constructor
  AlignTransform(){}

  /// Constructor without errors
  AlignTransform( const Translation & itranslation, 
				  const Translation & itranslationError,
                  const uint32_t & irawId ) :
	m_translation(itranslation),
	m_translationError(itranslationError)
	m_angles(0),m_angleErrors(0),
    m_rawId(irawId) {}
  
  /// Constructor with errors
  AlignTransform( const Translation & itranslation, 
				  const Translation & itranslationError,
				  const Angles & iangles,
				  const Angles & iangleErrors,
                  const uint32_t & irawId ) :
	m_translation(itranslation),
	m_translationError(itranslationError)
	m_angles(iangles),
	m_angleErrors(iangleErrors),
    m_rawId(irawId) {}

  const Translation & translation() const { return m_translation; }
  const Translation & translationError() const { return m_translationError; }
  const Angles & angles() const { return m_angles; }
  const Angles & angleErrors() const { return m_angleErrors; }
  const uint32_t & rawId() const { return m_rawId; }

private:

  Translation m_translation;
  Translation m_translationError;
  Angles      m_angles;
  Angles      m_angleErrors;
  uint32_t    m_rawId;


};
#endif //AlignTransform_H
