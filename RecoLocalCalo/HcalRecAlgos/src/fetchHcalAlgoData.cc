#include "RecoLocalCalo/HcalRecAlgos/interface/fetchHcalAlgoData.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Includes for AbsHcalAlgoData descendants
// and their corresponding records
#include "CondFormats/HcalObjects/interface/HFPhase1PMTParams.h"
#include "CondFormats/DataRecord/interface/HFPhase1PMTParamsRcd.h"

namespace {
    // Class Data must inherit from AbsHcalAlgoData
    // and must have a copy constructor. This function
    // returns an object allocated on the heap.
    template <class Data, class Record>
    Data* fetchHcalAlgoDataHelper(const edm::EventSetup& es)
    {
        edm::ESHandle<Data> p;
        es.get<Record>().get(p);
        return new Data(*p.product());
    }
}

std::unique_ptr<AbsHcalAlgoData>
fetchHcalAlgoData(const std::string& className, const edm::EventSetup& es)
{
    AbsHcalAlgoData* data = nullptr;

    // Compare with possibe class names
    //
    if (className == "HFPhase1PMTParams")
        data = fetchHcalAlgoDataHelper<HFPhase1PMTParams, HFPhase1PMTParamsRcd>(es);
    
    return std::unique_ptr<AbsHcalAlgoData>(data);
}
