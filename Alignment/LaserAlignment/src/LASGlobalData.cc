
#ifndef __LASGLOBALDATA_C
#define __LASGLOBALDATA_C

#include "Alignment/LaserAlignment/interface/LASGlobalData.h"




template <class T>
LASGlobalData<T>::LASGlobalData() {
  ///
  /// def constructor
  ///

  Init();

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





///
/// whatever initialization 
/// is needed
///
template <class T>
void LASGlobalData<T>::Init( void ) {

  // create TEC+ subdetector "multi"-vector of T
  tecPlusData.resize( 2 ); // create ring4 and ring6
  for( unsigned int ring = 0; ring < tecPlusData.size(); ++ring ) {
    tecPlusData.at( ring ).resize( 8 ); // create 8 beams for each ring
    for( unsigned int beam = 0; beam < tecPlusData.at( ring ).size(); ++beam ) {
      tecPlusData.at( ring ).at( beam ).resize( 9 ); // create 9 disks for each beam
    }
  }

  // same for TEC-
  tecMinusData.resize( 2 ); // create ring4 and ring6
  for( unsigned int ring = 0; ring < tecMinusData.size(); ++ring ) {
    tecMinusData.at( ring ).resize( 8 ); // create 8 beams for each ring
    for( unsigned int beam = 0; beam < tecMinusData.at( ring ).size(); ++beam ) {
      tecMinusData.at( ring ).at( beam ).resize( 9 ); // create 9 disks for each beam
    }
  }
  
  // same for TEC+ AT
  tecPlusATData.resize( 8 ); // create 8 beams
  for( unsigned int beam = 0; beam < tecPlusATData.size(); ++beam ) {
    tecPlusATData.at( beam ).resize( 5 ); // five TEC disks hit by each AT beam
  }

  // same for TEC- AT
  tecMinusATData.resize( 8 ); // create 8 beams
  for( unsigned int beam = 0; beam < tecMinusATData.size(); ++beam ) {
    tecMinusATData.at( beam ).resize( 5 ); // five TEC disks hit by each AT beam
  }

  // same for TIB..
  tibData.resize( 8 ); // create 8 beams
  for( unsigned int beam = 0; beam < tibData.size(); ++ beam ) {
    tibData.at( beam ).resize( 6 ); // six TIB modules hit by each beam
  }

  // ..and for TOB
  tobData.resize( 8 ); // create 8 beams
  for( unsigned int beam = 0; beam < tobData.size(); ++ beam ) {
    tobData.at( beam ).resize( 6 ); // six TOB modules hit by each beam
  }




}

#endif
