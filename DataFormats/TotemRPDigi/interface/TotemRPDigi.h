/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_TotemRPDigi_interface_TotemRPDigi_h
#define DataFormats_TotemRPDigi_interface_TotemRPDigi_h

#include "DataFormats/TotemRPDetId/interface/TotemRPIdTypes.h"

class TotemRPDigi
{
  public:
    TotemRPDigi(RPDetId det_id=0, unsigned short strip_no=0)
    {
      det_id_=det_id; 
      strip_no_=strip_no;
    };

    inline RPDetId GetDetId() const {return det_id_;}
    inline unsigned short GetStripNo() const {return strip_no_;}
  
  private:
    RPDetId det_id_;
    unsigned short strip_no_;
};


inline bool operator< (const TotemRPDigi& one, const TotemRPDigi& other)
{
  if(one.GetDetId() < other.GetDetId())
    return true;
  else if(one.GetDetId() == other.GetDetId())
    return one.GetStripNo() < other.GetStripNo();
  else 
    return false;
}

#endif
