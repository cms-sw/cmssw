#ifndef DataFormats_RPCDigi_RPCRawSynchro_H
#define DataFormats_RPCDigi_RPCRawSynchro_H

#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include <vector>

namespace RPCRawSynchro {

typedef std::vector<std::pair<LinkBoardElectronicIndex, int>> ProdItem;
}

#endif
