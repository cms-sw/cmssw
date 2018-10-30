#ifndef DETECTOR_DESCRIPTION_CORE_DD_POS_DATA_H
#define DETECTOR_DESCRIPTION_CORE_DD_POS_DATA_H

#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

//! Relative position of a child-volume inside a parent-volume
/**
  simple struct to collect information concerning the relative
  position of a child inside its parent.
*/
struct DDPosData
{
  //! Creates a new relative position
  /** \arg \c t relative translation std::vector
      \arg \c r relative rotation matrix
      \arg \c c copy number
      
      Users normally don't create DDPosData themselves. They get read access
      to relative positionings via DDPosData in DDCompactView.
  */    
  DDPosData(const DDTranslation & t, const DDRotation& r, int c, const DDDivision * d = nullptr )
   : trans_(t), rot_(r), copyno_(c), div_(d)
   {} 
  DDPosData() = delete;  
  DDPosData & operator=(const DDPosData &) = delete;

  const DDTranslation & translation() const { return trans_; }
  const DDTranslation & trans() const { return trans_; }

  const DDRotationMatrix & rotation() const { return rot_.rotation(); }
  const DDRotationMatrix & rot() const { return rot_.rotation(); }
  const DDRotation & ddrot() const { return rot_; }
  int copyno() const { return copyno_; }

  const DDDivision & div() const { return *div_; }
  const DDDivision & division() const { return *div_; }

private:
  DDTranslation trans_; /**< relative translation std::vector */
  DDRotation rot_; /**< relative rotation matrix */
  int copyno_; /**< copy number */
  const DDDivision * div_; /**< provides original division that created this pos */
};

#endif
