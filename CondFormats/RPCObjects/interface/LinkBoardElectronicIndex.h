#ifndef CondFormatsRPCObjectsLinkBoardElectronicIndex_H
#define CondFormatsRPCObjectsLinkBoardElectronicIndex_H

/* \class LinkBoardElectronicIndex
 * chamber location describtion as given by RawData, naming follows DB
 */

#include <string>

struct LinkBoardElectronicIndex {
  int dccId;
  int dccInputChannelNum;
  int tbLinkInputNum;
  int lbNumInLink;
  std::string print(int depth = 0) const;
};
#endif
