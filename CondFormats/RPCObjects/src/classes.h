#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "CondFormats/RPCObjects/interface/FebSpec.h"
template std::vector<ChamberStripSpec>::iterator;
template std::vector<ChamberStripSpec>::const_iterator;

#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
template std::vector<FebSpec>::iterator;
template std::vector<FebSpec>::const_iterator;

#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
template std::vector<LinkBoardSpec>::iterator;
template std::vector<LinkBoardSpec>::const_iterator;

#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
template std::vector<LinkConnSpec>::iterator;
template std::vector<LinkConnSpec>::const_iterator;

#include "CondFormats/RPCObjects/interface/DccSpec.h"
template std::vector<TriggerBoardSpec>::iterator;
template std::vector<TriggerBoardSpec>::const_iterator;

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
template std::map<int, DccSpec>::iterator;
template std::map<int, DccSpec>::const_iterator;


