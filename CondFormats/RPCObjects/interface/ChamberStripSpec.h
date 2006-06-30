#ifndef CondFormatsRPCObjectsChamberStripSpec_H
#define CondFormatsRPCObjectsChamberStripSpec_H

/** \class ChamberStripSpec
 * RPC strip specification for readout decoding
 */

struct ChamberStripSpec {
  int cablePinNumber;
  int chamberStripNumber;
  
  /// debug printout
  void print( int depth = 0) const;
};

#endif
