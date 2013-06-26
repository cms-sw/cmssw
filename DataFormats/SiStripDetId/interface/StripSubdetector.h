#ifndef DataFormats_SiStripDetId_StripSubdetector_H
#define DataFormats_SiStripDetId_StripSubdetector_H

/** 
 *  Enumeration for Strip Tracker Subdetectors
 *
 */


#include "DataFormats/DetId/interface/DetId.h"

class StripSubdetector :public DetId { 
 public:
 enum SubDetector { TIB=3,TID=4,TOB=5,TEC=6 };
 
 /** Constructor from a raw value */
 StripSubdetector(uint32_t rawid) : DetId(rawid) {}
 /**Construct from generic DetId */
 StripSubdetector(const DetId& id) : DetId(id) {}
 
 
 /// glued
 /**
  * glued() = 0 it's not a glued module
  * glued() != 0 it's a glued module
  */
 unsigned int glued() const
   {
     if(((id_>>sterStartBit_)& sterMask_) == 1){
       return (id_ -1);
     }else if(((id_>>sterStartBit_)& sterMask_) == 2){
       return (id_ -2);
     }else{
       return 0;
     }
   }
 
 /// stereo 
 /**
  * stereo() = 0 it's not a stereo module
  * stereo() = 1 it's a stereo module
  */
 unsigned int stereo() const 
   {
     if(((id_>>sterStartBit_)& sterMask_)==1){
       return ((id_>>sterStartBit_)& sterMask_);
     }else{
       return 0;
     }
   }
 
 /**
  * If the DetId identify a glued module return 
  * the DetId of your partner otherwise return 0
  */
 
 unsigned int partnerDetId() const
   {
     if(((id_>>sterStartBit_)& sterMask_)==1){
       return (id_ + 1);
     }else if(((id_>>sterStartBit_)& sterMask_)==2){
       return (id_ - 1);
     }else{
       return 0;
     }
   }
 
 private:
 static const unsigned int detStartBit_=       2;
 static const unsigned int sterStartBit_=      0;
 
 static const unsigned int detMask_=           0x3;
 static const unsigned int sterMask_=          0x3;
 
 
};


#endif
