
#ifndef __LASGLOBALDATA_H
#define __LASGLOBALDATA_H

#include<vector>
#include<iostream>
#include "TNamed.h"

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
class LASGlobalData : public TNamed
{

 public:
  // enums not in use so far...
  enum Subdetector { TECPLUS, TECMINUS, TIB, TOB };
  enum TecRing { RING4, RING6 };
  enum Beam { BEAM0, BEAM1, BEAM2, BEAM3, BEAM4, BEAM5, BEAM6, BEAM7 };
  enum TecDisk { DISK1, DISK2, DISK3, DISK4, DISK5, DISK6, DISK7, DISK8, DISK9 };
  enum TibTobPosition { MINUS3, MINUS2, MINUS1, PLUS1, PLUS2, PLUS3 };
  LASGlobalData();
  LASGlobalData(const T&);
  T& GetTECEntry( int subdetector, int tecRing, int beam, int tecDisk );
  T& GetTIBTOBEntry( int subdetector, int beam, int tibTobPosition );
  T& GetTEC2TECEntry( int subdetector, int beam, int tecDisk );
  void SetTECEntry( int subdetector, int tecRing, int beam, int tecDisk, T );
  void SetTIBTOBEntry( int subdetector, int beam, int tibTobPosition, T );
  void SetTEC2TECEntry( int subdetector, int beam, int tecDisk, T );
  //  LASGlobalData<T>& operator=( LASGlobalData<T>& );

 private:
  //void Init( void );
  void Init( const T& in=T());
  std::vector<std::vector<std::vector<T> > > tecPlusData; // ring<beam<disk<T>>>
  std::vector<std::vector<std::vector<T> > > tecMinusData; // ring<beam<disk<T>>>
  std::vector<std::vector<T> > tecPlusATData; // beam<disk<T>>
  std::vector<std::vector<T> > tecMinusATData; // beam<disk<T>>
  std::vector<std::vector<T> > tibData; // beam<pos<T>>
  std::vector<std::vector<T> > tobData; // beam<pos<T>>

  ClassDef( LASGlobalData, 2 );
};

// since this is a template
//#include "Alignment/LaserAlignment/src/LASGlobalData.cc"


template <class T>
LASGlobalData<T>::LASGlobalData() {
  ///
  /// def constructor
  ///

  Init();

}

template <class T>
LASGlobalData<T>::LASGlobalData(const T& in)
{
  Init(in);
}



///
/// get a tec entry from the container according to
/// subdetector, ring, beam and disk number
///
template <class T>
T& LASGlobalData<T>::GetTECEntry( int theDetector, int theRing, int theBeam, int theDisk ) {
  
  // do a range check first
  if( !( ( theDetector == 0 || theDetector == 1 ) &&        // TEC+ or TEC-
	 ( theRing == 0 || theRing == 1 )         &&        // ring4 or ring6
	 ( theBeam >= 0 && theBeam < 8 )          &&        // eight beams in a TEC
	 ( theDisk >= 0 && theDisk < 9 )             ) ) {  // disk1..disk9
    std::cerr << " [LASGlobalData::GetTECEntry] ** ERROR: illegal input coordinates:" << std::endl;
    std::cerr << "   detector " << theDetector << ", ring " << theRing << ", beam " << theBeam << ", disk " << theDisk << "." << std::endl;
    throw   "   Bailing out."; // @@@ REPLACE THIS BY cms::Exception (<FWCore/Utilities/interface/Exception.h> in 1_3_6)
  }
  else {
    if( theDetector == 0 ) return( tecPlusData.at( theRing ).at( theBeam ).at( theDisk ) );
    else return( tecMinusData.at( theRing ).at( theBeam ).at( theDisk ) );
  }

}




///
/// get a tib/tob entry from the container according to
/// subdetector, beam and position (z) number 
///
template <class T>
T& LASGlobalData<T>::GetTIBTOBEntry( int theDetector, int theBeam, int thePosition ) {

  // do a range check first
  if( !( ( theDetector == 2 || theDetector == 3 ) &&        // TIB or TOB
	 ( theBeam >= 0 && theBeam < 8 )          &&        // there are eight AT beams
	 ( thePosition >= 0 && thePosition < 6 )     ) ) {  // z-pos -3 .. z-pos +3
    std::cerr << " [LASGlobalData::GetTIBTOBEntry] ** ERROR: illegal coordinates:" << std::endl;
    std::cerr << "   detector " << theDetector << ", beam " << theBeam << ", position " << thePosition << "." << std::endl;
    throw   "   Bailing out."; // @@@ REPLACE THIS BY cms::Exception (<FWCore/Utilities/interface/Exception.h> in 1_3_6)
  }
  else {
    if( theDetector == 2 ) return( tibData.at( theBeam ).at( thePosition ) );
    else return( tobData.at( theBeam ).at( thePosition ) );
  }

}




///
/// get a tec AT entry (ring 4) from the container according to
/// subdetector, beam and disk number 
///
template <class T>
T& LASGlobalData<T>::GetTEC2TECEntry( int theDetector, int theBeam, int theDisk ) {

  // do a range check first
  if( !( ( theDetector == 0 || theDetector == 1 ) &&        // TEC+ or TEC-
	 ( theBeam >= 0 && theBeam < 8 )          &&        // eight AT beams in a TEC
	 ( theDisk >= 0 && theDisk < 6 )     ) ) {          // disk1...disk5 are hit by AT
    std::cerr << " [LASGlobalData::GetTEC2TECEntry] ** ERROR: illegal coordinates:" << std::endl;
    std::cerr << "   detector " << theDetector << ", beam " << theBeam << ", disk " << theDisk << "." << std::endl;
    throw   "   Bailing out."; // @@@ REPLACE THIS BY cms::Exception (<FWCore/Utilities/interface/Exception.h> in 1_3_6)
  }
  else {
    if( theDetector == 0 ) return( tecPlusATData.at( theBeam ).at( theDisk ) );
    else return( tecMinusATData.at( theBeam ).at( theDisk ) );
  }

}





///
/// set a tec entry int the container according to
/// subdetector, ring, beam and disk number
///
template <class T>
void LASGlobalData<T>::SetTECEntry( int theDetector, int theRing, int theBeam, int theDisk, T theEntry ) {
  
  // do a range check first
  if( !( ( theDetector == 0 || theDetector == 1 ) &&        // TEC+ or TEC-
	 ( theRing == 0 || theRing == 1 )         &&        // ring4 or ring6
	 ( theBeam >= 0 && theBeam < 8 )          &&        // eight beams in a TEC
	 ( theDisk >= 0 && theDisk < 9 )             ) ) {  // disk1..disk9
    std::cerr << " [LASGlobalData::SetTECEntry] ** ERROR: illegal coordinates:" << std::endl;
    std::cerr << "   detector " << theDetector << ", ring " << theRing << ", beam " << theBeam << ", disk " << theDisk << "." << std::endl;
    throw   "   Bailing out."; // @@@ REPLACE THIS BY cms::Exception (<FWCore/Utilities/interface/Exception.h> in 1_3_6)
  }
  else {
    if( theDetector == 0 ) tecPlusData.at( theRing ).at( theBeam ).at( theDisk ) = theEntry;
    else tecMinusData.at( theRing ).at( theBeam ).at( theDisk ) = theEntry;
  }

}





///
///  set a tib/tob entry in the container accord
///  subdetector, beam and position (z) number
///
template <class T>
void LASGlobalData<T>::SetTIBTOBEntry( int theDetector, int theBeam, int thePosition, T theEntry ) {

  // do a range check first
  if( !( ( theDetector == 2 || theDetector == 3 ) &&        // TIB or TOB
	 ( theBeam >= 0 && theBeam < 8 )          &&        // there are eight AT beams
	 ( thePosition >= 0 && thePosition < 6 )     ) ) {  // pos-3..pos+3
    std::cerr << " [LASGlobalData::SetTIBTOBEntry] ** ERROR: illegal coordinates:" << std::endl;
    std::cerr << "   detector " << theDetector << ", beam " << theBeam << ", position " << thePosition << "." << std::endl;
    throw   "   Bailing out."; // @@@ REPLACE THIS BY cms::Exception (<FWCore/Utilities/interface/Exception.h> in 1_3_6)
  }
  else {
    if( theDetector == 2 ) tibData.at( theBeam ).at( thePosition ) = theEntry;
    else tobData.at( theBeam ).at( thePosition ) = theEntry;
  }

}





///
/// set a tec AT entry (ring 4) in the container according to
/// subdetector, beam and disk number
///
template <class T>
void LASGlobalData<T>::SetTEC2TECEntry( int theDetector, int theBeam, int theDisk, T theEntry ) {

  // do a range check first
  if( !( ( theDetector == 0 || theDetector == 1 ) &&        // TEC+ or TEC-
	 ( theBeam >= 0 && theBeam < 8 )          &&        // eight beams in a TEC
	 ( theDisk >= 0 && theDisk < 6 )             ) ) {  // disk1..disk5 for TEC AT
    std::cerr << " [LASGlobalData::SetTEC2TECEntry] ** ERROR: illegal coordinates:" << std::endl;
    std::cerr << "   detector " << theDetector << ", beam " << theBeam << ", disk " << theDisk << "." << std::endl;
    throw   "   Bailing out."; // @@@ REPLACE THIS BY cms::Exception (<FWCore/Utilities/interface/Exception.h> in 1_3_6)
  }
  else {
    if( theDetector == 0 ) tecPlusATData.at( theBeam ).at( theDisk ) = theEntry;
    else tecMinusATData.at( theBeam ).at( theDisk ) = theEntry;
  }

}





// ///
// /// element wise assignment operator
// ///
// template <class T>
// LASGlobalData<T>& LASGlobalData<T>::operator=( LASGlobalData<T>& anotherGlobalData ) {

//   // TEC copy
//   for( int det = 0; det < 2; ++det ) {
//     for( int ring = 0; ring < 2; ++ring ) {
//       for( int beam = 0; beam < 8; ++ beam ) {
// 	for( int disk = 0; disk < 9; ++ disk ) {
// 	  this->SetTECEntry( det, ring, beam, disk, anotherGlobalData.GetTECEntry( det, ring, beam, disk ) );
// 	}
//       }
//     }
//   }

//   // TIBTOB copy
//   for( int det = 2; det < 4; ++det ) {
//     for( int beam = 0; beam < 8; ++ beam ) {
//       for( int pos = 0; pos < 6; ++ pos ) {
// 	this->SetTIBTOBEntry( det, beam, pos, anotherGlobalData.GetTIBTOBEntry( det, beam, pos ) );
//       }
//     }
//   }
  
//   // TEC2TEC copy
//   for( int det = 2; det < 4; ++det ) {
//     for( int beam = 0; beam < 8; ++ beam ) {
//       for( int disk = 0; disk < 9; ++ disk ) {
// 	this->SetTEC2TECEntry( det, beam, disk, anotherGlobalData.GetTEC2TECEntry( det, beam, disk ) );
//       }
//     }
//   }

// }



template <class T>
void LASGlobalData<T>::Init( const T& in ) {

  // create TEC+ subdetector "multi"-vector of T
  tecPlusData.resize( 2 ); // create ring4 and ring6
  for( unsigned int ring = 0; ring < tecPlusData.size(); ++ring ) {
    tecPlusData.at( ring ).resize( 8 ); // create 8 beams for each ring
    for( unsigned int beam = 0; beam < tecPlusData.at( ring ).size(); ++beam ) {
      tecPlusData.at( ring ).at( beam ).resize( 9 , in); // create 9 disks for each beam
    }
  }

  // same for TEC-
  tecMinusData.resize( 2 ); // create ring4 and ring6
  for( unsigned int ring = 0; ring < tecMinusData.size(); ++ring ) {
    tecMinusData.at( ring ).resize( 8 ); // create 8 beams for each ring
    for( unsigned int beam = 0; beam < tecMinusData.at( ring ).size(); ++beam ) {
      tecMinusData.at( ring ).at( beam ).resize( 9, in ); // create 9 disks for each beam
    }
  }
  
  // same for TEC+ AT
  tecPlusATData.resize( 8 ); // create 8 beams
  for( unsigned int beam = 0; beam < tecPlusATData.size(); ++beam ) {
    tecPlusATData.at( beam ).resize( 5, in ); // five TEC disks hit by each AT beam
  }

  // same for TEC- AT
  tecMinusATData.resize( 8 ); // create 8 beams
  for( unsigned int beam = 0; beam < tecMinusATData.size(); ++beam ) {
    tecMinusATData.at( beam ).resize( 5, in ); // five TEC disks hit by each AT beam
  }

  // same for TIB..
  tibData.resize( 8 ); // create 8 beams
  for( unsigned int beam = 0; beam < tibData.size(); ++ beam ) {
    tibData.at( beam ).resize( 6, in ); // six TIB modules hit by each beam
  }

  // ..and for TOB
  tobData.resize( 8 ); // create 8 beams
  for( unsigned int beam = 0; beam < tobData.size(); ++ beam ) {
    tobData.at( beam ).resize( 6, in ); // six TOB modules hit by each beam
  }
}


#endif
