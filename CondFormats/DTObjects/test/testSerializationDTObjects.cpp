#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"


int main()
{
    testSerialization<DTCCBConfig>();
    testSerialization<DTCCBId>();
    testSerialization<DTConfigKey>();
    testSerialization<DTDeadFlag>();
    testSerialization<DTDeadFlagData>();
    testSerialization<DTDeadFlagId>();
    testSerialization<DTHVStatus>();
    testSerialization<DTHVStatusData>();
    testSerialization<DTHVStatusId>();
    testSerialization<DTKeyedConfig>();
    testSerialization<DTLVStatus>();
    testSerialization<DTLVStatusData>();
    testSerialization<DTLVStatusId>();
    testSerialization<DTMtime>();
    testSerialization<DTMtimeData>();
    testSerialization<DTMtimeId>();
    testSerialization<DTPerformance>();
    testSerialization<DTPerformanceData>();
    testSerialization<DTPerformanceId>();
    testSerialization<DTRangeT0>();
    testSerialization<DTRangeT0Data>();
    testSerialization<DTRangeT0Id>();
    testSerialization<DTReadOutGeometryLink>();
    testSerialization<DTReadOutMapping>();
    testSerialization<DTRecoUncertainties>();
    testSerialization<DTRecoConditions>();
    testSerialization<DTStatusFlag>();
    testSerialization<DTStatusFlagData>();
    testSerialization<DTStatusFlagId>();
    testSerialization<DTT0>();
    testSerialization<DTT0Data>();
    testSerialization<DTTPGParameters>();
    testSerialization<DTTPGParametersData>();
    testSerialization<DTTPGParametersId>();
    testSerialization<DTTtrig>();
    testSerialization<DTTtrigData>();
    testSerialization<DTTtrigId>();
    testSerialization<std::pair<DTCCBId,int>>();
    testSerialization<std::pair<DTDeadFlagId,DTDeadFlagData>>();
    testSerialization<std::pair<DTHVStatusId,DTHVStatusData>>();
    testSerialization<std::pair<DTLVStatusId,DTLVStatusData>>();
    testSerialization<std::pair<DTMtimeId,DTMtimeData>>();
    testSerialization<std::pair<DTPerformanceId,DTPerformanceData>>();
    testSerialization<std::pair<DTRangeT0Id,DTRangeT0Data>>();
    testSerialization<std::pair<DTStatusFlagId,DTStatusFlagData>>();
    testSerialization<std::pair<DTTPGParametersId,DTTPGParametersData>>();
    testSerialization<std::pair<DTTtrigId,DTTtrigData>>();
    testSerialization<std::vector<std::pair<DTMtimeId,DTMtimeData>>>();
    testSerialization<std::vector<std::pair<DTStatusFlagId,DTStatusFlagData>>>();
    testSerialization<std::vector<std::pair<DTTtrigId,DTTtrigData>>>();
    testSerialization<std::vector<DTConfigKey>>();
    testSerialization<std::vector<DTReadOutGeometryLink>>();
    testSerialization<std::vector<DTT0Data>>();
    testSerialization<std::vector<std::pair<DTCCBId,int>>>();
    testSerialization<std::vector<std::pair<DTDeadFlagId,DTDeadFlagData>>>();
    testSerialization<std::vector<std::pair<DTHVStatusId,DTHVStatusData>>>();
    testSerialization<std::vector<std::pair<DTLVStatusId,DTLVStatusData>>>();
    testSerialization<std::vector<std::pair<DTPerformanceId,DTPerformanceData>>>();
    testSerialization<std::vector<std::pair<DTRangeT0Id,DTRangeT0Data>>>();
    testSerialization<std::vector<std::pair<DTTPGParametersId,DTTPGParametersData>>>();

    return 0;
}
