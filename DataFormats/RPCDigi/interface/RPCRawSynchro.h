#ifndef DataFormats_RPCDigi_RPCRawSynchro_H
#define DataFormats_RPCDigi_RPCRawSynchro_H

#include <vector>
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"

namespace RPCRawSynchro {

  typedef std::vector<std::pair<LinkBoardElectronicIndex, int> > ProdItem;
}

#endif
