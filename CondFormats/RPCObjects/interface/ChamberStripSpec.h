#ifndef CondFormatsRPCObjectsChamberStripSpec_H
#define CondFormatsRPCObjectsChamberStripSpec_H

/** \class ChamberStripSpec
 * RPC strip specification for readout decoding
 */

struct ChamberStripSpec {
  int cablePinNumber;
  int chamberStripNumber;
  int cmsStripNumber;
  
  /// debug printout
  void print( int depth = 0) const;
};

#endif
