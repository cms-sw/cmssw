#ifndef AlignTransform_H
#define AlignTransform_H
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CLHEP/Vector/EulerAngles.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Geometry/Transform3D.h"

#include "CondFormats/Alignment/interface/Definitions.h"

/// Class holding data for an Alignment transformation
/// It contains the raw detector id, its global position and global orientation.
/// It is optimized for storage.
class  AlignTransform 
{
public:
  typedef CLHEP::HepEulerAngles EulerAngles;
  typedef CLHEP::Hep3Vector     Translation;
  typedef CLHEP::HepRotation    Rotation;
  typedef HepGeom::Transform3D Transform;

  /// Default constructor
  AlignTransform(){}

  /// Constructor from Euler angles
  AlignTransform( const Translation & itranslation, 
		  const EulerAngles & ieulerAngles,
                  align::ID irawId ) :
    m_translation(itranslation),
    m_eulerAngles(ieulerAngles),
    m_rawId(irawId) {}
  
  /// Constructor from Rotation
  AlignTransform( const Translation & itranslation, 
		  const Rotation    & irotation,
                  align::ID irawId ) :
    m_translation(itranslation),
    m_eulerAngles(irotation.eulerAngles()),
    m_rawId(irawId) {}

  const Translation & translation() const { return m_translation; }
  /// Do not expose Euler angles since we may change its type later
  //   const EulerAngles & eulerAngles() const { return m_eulerAngles; }
  align::ID rawId() const { return m_rawId; }

  Rotation rotation() const 
  { //std::cout<<"Inside aligntransform::rotation() with id="<<std::hex<<m_rawId<<std::dec<<std::endl ;
     //std::cout<<" for e.a.="<<m_eulerAngles<<std::endl;
     return Rotation(m_eulerAngles); }

  Transform transform() const { return Transform( rotation(), translation() ) ; }  

  // Implemented so this can be sorted by rawId
  const bool operator < ( const AlignTransform & other ) const
  {
        return ( m_rawId < other.rawId() );
  }
      
 private:

  Translation m_translation;
  EulerAngles m_eulerAngles;
  align::ID   m_rawId;



 COND_SERIALIZABLE;
};
#endif //AlignTransform_H
