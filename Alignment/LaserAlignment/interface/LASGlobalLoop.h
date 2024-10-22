
#ifndef __LASGLOBALLOOP_H
#define __LASGLOBALLOOP_H

#include <iostream>

///
/// helper class for looping over LASGlobalData objects
/// (si-strip module loops). Use exactly as:
/// \code
/// LASGlobalLoop theLoop;
/// int det = 0, ring = 0, beam = 0, disk = 0;
/// do {
///   // det,ring,beam,disk will loop the full TEC+,TEC-
/// } while ( loop.TECLoop( det, ring, beam, disk ) );
/// int pos = 0;
/// det = 2; // set subdetector to TIB
/// beam = 0;
/// do {
///   // dto.
/// } while( loop.TIBTOBLoop( det, beam, disk ) );
/// \endcode
///
class LASGlobalLoop {
public:
  LASGlobalLoop();
  bool TECLoop(int&, int&, int&, int&) const;
  bool TIBTOBLoop(int&, int&, int&) const;
  bool TEC2TECLoop(int&, int&, int&) const;
};

#endif
