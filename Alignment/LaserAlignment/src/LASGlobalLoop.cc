
#include "Alignment/LaserAlignment/interface/LASGlobalLoop.h"


///
///
///
LASGlobalLoop::LASGlobalLoop() {
}





///
/// full TEC loop (both endcaps)
/// with starting values given by parameter values
///
bool LASGlobalLoop::TECLoop( int& subdet, int& ring, int& beam, int& disk ) const {

  if( subdet > 1 ) {
    std::cerr << " [LASGlobalLoop::TECLoop] ** ERROR: Endcap loop running on TIB/TOB (subdetector > 1)" << std::endl;
    throw 1;
  }

  ++disk;

  if( disk == 9 ) { 
    ++beam;
    disk = 0;

    if( beam == 8 ) {
      ++ring;
      beam = 0;

      if( ring == 2 ) {
	++subdet;
	ring = 0;

	if( subdet == 2 ) return false;

      }
    }
  }

  return true;

}





///
/// full TIB+TOB loop
/// with starting values given by parameter values
///
bool LASGlobalLoop::TIBTOBLoop( int& subdet, int& beam, int& position ) const {

  if( subdet < 2 ) {
    std::cerr << " [LASGlobalLoop::TIBTOBLoop] ** ERROR: Barrel loop running on TEC (subdetector < 2)" << std::endl;
    throw 1;
  }

  ++position;

  if( position == 6 ) {
    ++beam;
    position = 0;

    if( beam == 8 ) {
      ++subdet;
      beam = 0;
      
      if( subdet == 4 ) return false;

    }
  }

  return true;

}





///
/// full TEC AT loop
/// with starting values given by parameter values
///
bool LASGlobalLoop::TEC2TECLoop( int& subdet, int& beam, int& disk ) const {

  if( subdet > 1 ) {
    std::cerr << " [LASGlobalLoop::TEC2TECLoop] ** ERROR: TEC loop running on TIB/TOB (subdetector > 1)" << std::endl;
    throw 1;
  }

  ++disk;

  if( disk == 5 ) {
    ++beam;
    disk = 0;

    if( beam == 8 ) {
      ++subdet;
      beam = 0;
      
      if( subdet == 2 ) return false;

    }
  }

  return true;

}
