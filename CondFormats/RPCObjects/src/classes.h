#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/RPCdeteIndex.h"
#include "CondFormats/RPCObjects/interface/RPCelecIndex.h"

namespace{
  std::map<RPCdeteIndex, RPCelecIndex> cmap;
}
namespace{
  std::map<RPCelecIndex, RPCdeteIndex> dmap;
}
