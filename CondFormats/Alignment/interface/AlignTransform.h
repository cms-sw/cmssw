#ifndef AlignTransform_H
#define AlignTransform_H
#include "CLHEP/Vector/EulerAngles.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/ThreeVector.h"


// a Class holding data for an Allignment transformation
/** it is optimized for storage
 **/
class  AlignTransform {
public:
  typedef CLHEP::HepEulerAngles EulerAngles;
  typedef CLHEP::Hep3Vector ThreeVector;
  typedef CLHEP::HepRotation Rotation;
  typedef ThreeVector Translation;


  AlignTransform(){}
  AlignTransform( const Translation & itranslation, 
		  const EulerAngles & ieulerAngles) :
    m_translation(itranslation),
    m_eulerAngles(ieulerAngles){}
  AlignTransform( const Translation & itranslation, 
		  const Rotation & irotation) :
    m_translation(itranslation),
    m_eulerAngles(irotation.eulerAngles()){}
  
  const Translation & translation() const { return m_translation; }
  const EulerAngles & eulerAngles() const { return m_eulerAngles; }
  Rotation rotation() const { return Rotation(m_eulerAngles); }

private:

  Translation m_translation;
  EulerAngles m_eulerAngles;


};
#endif //AlignTransform_H
