#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"

// Declaration of the iterator (necessary for the generation of the dictionary)
template std::vector<DTReadOutGeometryLink>::iterator;
template std::vector<DTReadOutGeometryLink>::const_iterator;
template std::vector<DTCellT0Data>::iterator;
template std::vector<DTCellT0Data>::const_iterator;
template std::vector<DTSLTtrigData>::iterator;
template std::vector<DTSLTtrigData>::const_iterator;
template std::vector<DTSLMtimeData>::iterator;
template std::vector<DTSLMtimeData>::const_iterator;


