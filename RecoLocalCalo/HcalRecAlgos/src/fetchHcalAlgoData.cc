#include "RecoLocalCalo/HcalRecAlgos/interface/fetchHcalAlgoData.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Includes for AbsHcalAlgoData descendants
// and their corresponding records

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
    AbsHcalAlgoData* data = 0;
    //
    // Compare with possibe class names
    // if (className == "MyHcalAlgoData")
    //     data = fetchHcalAlgoDataHelper<MyHcalAlgoData, MyHcalAlgoDataRcd>(es);
    // else if (className == "OtherHcalAlgoData")
    //     ...;
    //
    return std::unique_ptr<AbsHcalAlgoData>(data);
}
