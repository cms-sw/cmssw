#ifndef CondFormatsRPCObjectsChamberStripSpec_H
#define CondFormatsRPCObjectsChamberStripSpec_H

/** \class ChamberStripSpec
 * RPC strip specification for readout decoding
 */

struct ChamberStripSpec {
  char cablePinNumber;
  char chamberStripNumber;
  
  /// debug printout
  void print( int depth = 0) const;
};

#endif
