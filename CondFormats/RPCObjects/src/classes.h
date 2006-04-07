#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/RPCdeteIndex.h"
#include "CondFormats/RPCObjects/interface/RPCelecIndex.h"

// Declaration of the iterator (necessary for the generation of the dictionary)
template std::vector<RPCReadOutLink>::iterator;
template std::vector<RPCReadOutLink>::const_iterator;
template std::map<RPCdeteIndex, RPCelecIndex>::iterator;
template std::map<RPCdeteIndex, RPCelecIndex>::const_iterator;
template std::map<RPCelecIndex, RPCdeteIndex>::iterator;
template std::map<RPCelecIndex, RPCdeteIndex>::const_iterator;
