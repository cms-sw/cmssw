#ifndef CondFormatsRPCObjectsChamberRawDataSpec_H
#define CondFormatsRPCObjectsChamberRawDataSpec_H

/* \class ChamberRawDataSpec
 * chamber location describtion as given by RawData, naming follows DB
 */

#include <string>

struct ChamberRawDataSpec {
  int dccId;
  int dccInputChannelNum; 
  int tbLinkInputNum;
  int lbNumInLink; 
  std::string print( int depth = 0) const;
};
#endif
