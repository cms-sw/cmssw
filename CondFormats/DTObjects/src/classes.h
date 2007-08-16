#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DTObjects/interface/DTRangeT0.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DTObjects/interface/DTPerformance.h"

namespace {
  std::map<DTT0Id,DTT0Data,DTT0Compare> t0Map;
  std::map<DTTtrigId,DTTtrigData,DTTtrigCompare> tTrigMap;
  std::map<DTMtimeId,DTMtimeData,DTMtimeCompare> mTimeMap;
  std::map<DTRangeT0Id,DTRangeT0Data,DTRangeT0Compare> rangeT0Map;
  std::map<DTStatusFlagId,DTStatusFlagData,DTStatusFlagCompare> statusFlagMap;
  std::map<DTDeadFlagId,DTDeadFlagData,DTDeadFlagCompare> deadFlagMap;
  std::map<DTPerformanceId,DTPerformanceData,DTPerformanceCompare> performanceMap;
}

/*
// Declaration of the iterator (necessary for the generation of the dictionary)
template std::vector<DTReadOutGeometryLink>::iterator;
template std::vector<DTReadOutGeometryLink>::const_iterator;
//template std::vector<DTCellT0Data>::iterator;
//template std::vector<DTCellT0Data>::const_iterator;
template std::map<DTT0Id,DTT0Data,DTT0Compare>::iterator;
template std::map<DTT0Id,DTT0Data,DTT0Compare>::const_iterator;
//template std::vector<DTSLTtrigData>::iterator;
//template std::vector<DTSLTtrigData>::const_iterator;
template std::map<DTTtrigId,DTTtrigData,DTTtrigCompare>::iterator;
template std::map<DTTtrigId,DTTtrigData,DTTtrigCompare>::const_iterator;
//template std::vector<DTSLMtimeData>::iterator;
//template std::vector<DTSLMtimeData>::const_iterator;
template std::map<DTMtimeId,DTMtimeData,DTMtimeCompare>::iterator;
template std::map<DTMtimeId,DTMtimeData,DTMtimeCompare>::const_iterator;
//template std::vector<DTSLRangeT0Data>::iterator;
//template std::vector<DTSLRangeT0Data>::const_iterator;
template std::map<DTRangeT0Id,DTRangeT0Data,DTRangeT0Compare>::iterator;
template std::map<DTRangeT0Id,DTRangeT0Data,DTRangeT0Compare>::const_iterator;
//template std::vector<DTCellStatusFlagData>::iterator;
//template std::vector<DTCellStatusFlagData>::const_iterator;
template std::map<DTStatusFlagId,DTStatusFlagData,
                  DTStatusFlagCompare>::iterator;
template std::map<DTStatusFlagId,DTStatusFlagData,
                  DTStatusFlagCompare>::const_iterator;
template std::map<DTDeadFlagId,DTDeadFlagData,
                  DTDeadFlagCompare>::iterator;
template std::map<DTDeadFlagId,DTDeadFlagData,
                  DTDeadFlagCompare>::const_iterator;
template std::map<DTPerformanceId,DTPerformanceData,DTPerformanceCompare>::iterator;
template std::map<DTPerformanceId,DTPerformanceData,DTPerformanceCompare>::const_iterator;
*/

