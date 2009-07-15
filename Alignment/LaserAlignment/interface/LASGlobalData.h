
#ifndef __LASGLOBALDATA_H
#define __LASGLOBALDATA_H

#include<vector>
#include<iostream>

///
/// Container class for storing and easy access to global LAS data
///
/// There is one entry of type T for each LAS module, e.g. beam profiles, position, name, ...
/// All identifiers (beam,subdetector,position,...) start with index 0. Note that some ring 4
/// TEC modules are hit by either TEC internal as well as by AT beams and are therefore 
/// considered twice in the container (once in tec<X>Data and once in tec<X>ATData).
/// Do not instantiate this class with bool.
///
/// Short LAS geometry reminder:<BR>
/// <UL>
/// <LI>TEC internal alignment:<BR>
///   8 beams each hit ring 6 modules on 9 disks per endcap<BR>
///   8 beams each hit ring 4 modules on 9 disks per endcap<BR>
/// <LI>Barrel AT alignment:<BR>
///   8 AT beams each hit 6 TIb and 6 TOB modules<BR>
/// <LI>TEC AT (TEC2TEC inter-) alignment:<BR>
///   8 barrel (AT) beams each hit 5 ring 4 modules per endcap<BR>
/// </UL>
///
template <class T>
class LASGlobalData {

 public:
  // enums not in use so far...
  enum Subdetector { TECPLUS, TECMINUS, TIB, TOB };
  enum TecRing { RING4, RING6 };
  enum Beam { BEAM0, BEAM1, BEAM2, BEAM3, BEAM4, BEAM5, BEAM6, BEAM7 };
  enum TecDisk { DISK1, DISK2, DISK3, DISK4, DISK5, DISK6, DISK7, DISK8, DISK9 };
  enum TibTobPosition { MINUS3, MINUS2, MINUS1, PLUS1, PLUS2, PLUS3 };
  LASGlobalData();
  T& GetTECEntry( int subdetector, int tecRing, int beam, int tecDisk );
  T& GetTIBTOBEntry( int subdetector, int beam, int tibTobPosition );
  T& GetTEC2TECEntry( int subdetector, int beam, int tecDisk );
  void SetTECEntry( int subdetector, int tecRing, int beam, int tecDisk, T );
  void SetTIBTOBEntry( int subdetector, int beam, int tibTobPosition, T );
  void SetTEC2TECEntry( int subdetector, int beam, int tecDisk, T );
  //  LASGlobalData<T>& operator=( LASGlobalData<T>& );

 private:
  void Init( void );
  std::vector<std::vector<std::vector<T> > > tecPlusData; // ring<beam<disk<T>>>
  std::vector<std::vector<std::vector<T> > > tecMinusData; // ring<beam<disk<T>>>
  std::vector<std::vector<T> > tecPlusATData; // beam<disk<T>>
  std::vector<std::vector<T> > tecMinusATData; // beam<disk<T>>
  std::vector<std::vector<T> > tibData; // beam<pos<T>>
  std::vector<std::vector<T> > tobData; // beam<pos<T>>

};

// since this is a template
#include "Alignment/LaserAlignment/src/LASGlobalData.cc"

#endif
