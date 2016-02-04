#ifndef DDPosData_h
#define DDPosData_h

#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
//#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

//! Relative position of a child-volume inside a parent-volume
/**
  simple struct to collect information concerning the relative
  position of a child inside its parent.
  
  \a replication is currently unused!
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
  DDPosData(const DDTranslation & t, const DDRotation& r, int c, const DDDivision * d = NULL )  //(mec:2007-06-07) tried = 0 when i did the delete in destructor... no help/difference. 
   : trans_(t), rot_(r), replication_(0), copyno_(c), div_(d)
   {
     //if (!rot_.rotation()) throw DDException("rotation not defined: [" + rot_.ns() + ":" + rot_.name() +"]" ); 
   } 

  /* Prior to this attempt do only delete div_ if it existed, we had less lost memory (mec:2007-06-07)   
     ~DDPosData() { 
     // delete &trans_; 
     if ( div_ == 0 ) delete div_; 
     } */
  const DDTranslation & translation() const { return trans_; }
  const DDTranslation & trans() const { return trans_; }
  
  const DDRotationMatrix & rotation() const { return *(rot_.rotation()); }
  const DDRotationMatrix & rot() const { return *(rot_.rotation()); }

  const DDDivision & div() const { return *div_; }
  const DDDivision & division() const { return *div_; }
  
  //const DDTranslation & trans_; /**< relative translation std::vector */
  DDTranslation trans_; /**< relative translation std::vector */
  
  DDRotation rot_; /**< relative rotation matrix */
  //FIXME: DDPosData: replication_ provide a design!
  void * replication_; /**< currently \b not used! */
  int copyno_; /**< copy number */
  const DDDivision * div_; /**< provides original division that created this pos */

private:
  DDPosData();  
  DDPosData & operator=(const DDPosData &);
};
#endif
