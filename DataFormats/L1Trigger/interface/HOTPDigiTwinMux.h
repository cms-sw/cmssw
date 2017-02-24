#ifndef HOTPDIGI_TWINMUX_H
#define HOTPDIGI_TWINMUX_H

#include <boost/cstdint.hpp>
#include <ostream>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

/** \class HOTPDigiTwinMux
  *  Simple container packer/unpacker for HO TriggerPrimittive in TwinMUX
  *  Trigger Primitive from HO HTR
  *
  *  \author Saxena, Pooja - DESY
  */

class HOTPDigiTwinMux {
 public:
  typedef HcalDetId key_type; /// For the sorted collection

  HOTPDigiTwinMux() {theTP_HO=0;}
  HOTPDigiTwinMux(uint64_t data) {theTP_HO = data;}
  HOTPDigiTwinMux(int ieta, int iphi, int bx, int mip, int validbit, int wheel, int sector, int index, int link);

  const HcalDetId id() const { return HcalDetId(HcalOuter,ieta(),iphi(),4); }
  
  /// get raw packed HO 
  uint64_t raw() const {return theTP_HO; }
  
  /// get the raw ieta value
  int raw_ieta() const {return theTP_HO&0x1F; }
  
  /// get the sign of ieta (int: +/- 1)
  int ieta_sign() const {return ( (theTP_HO&0x10)?(-1):(+1)); }

  /// get the absolute value of ieta
  int ieta_abs() const {return (theTP_HO&0x000F); }
  
  /// get the signed ieta value
  int ieta() const {return (ieta_abs() * ieta_sign()); }

  /// get the raw iphi value
  int iphi() const {return (theTP_HO>>5)&0x007F; }
  
  /// get the bx()
  int bx_abs() const {return (theTP_HO>>12)&0x1; }

  /// get the bx sign
  //  int bx_sign() const {return ( ( (theTP_HO>>13)&0x2000) ?(-1):(+1)); }
  int bx_sign() const {return ( ( (theTP_HO>>13)&0x1) ?(-1):(+1)); }

  //get bx
  int bx() const {return  (bx_abs() * bx_sign()); } 
  
  /// get the mip value
  int mip() const {return (theTP_HO>>14)&0x1; }

  /// get the valid bit
  int validbit() const {return (theTP_HO>>15)&0x1; } //MIP consistency check with HO FEDs

  /// get the raw wheel value
  int raw_wheel() const {return (theTP_HO>>16)&0x7; }
  
  /// get the sign of wheel (int: +/- 1)             
  int wheel_sign() const {return ( ( (theTP_HO>>18)&0x1) ?(-1):(+1)); }

  /// get the absolute value of wheel                                             
  int wheel_abs() const {return (theTP_HO>>16)&0x03; }
  
  /// get the signed wheel value                                                
  int wheel() const {return (wheel_abs() * wheel_sign()); }

  /// get the sector value
  int sector() const {return (theTP_HO>>19)&0xF; }
  
  /// get the index
  int index() const {return (theTP_HO>>23)&0x1F; } //channel index in Twinmux protocal 

  /// get the link value
  int link() const {return (theTP_HO>>28)&0x3; } //two link for all HO wheels

  static const int HO_SECTOR_MAX = 12;

 private:
  uint64_t theTP_HO;
};

std::ostream& operator<<(std::ostream&, const HOTPDigiTwinMux&);

#endif


  
