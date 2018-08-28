#ifndef RecoEgamma_EgammaTools_Utils_H
#define RecoEgamma_EgammaTools_Utils_H

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include <memory>

template<typename ValueType, class HandleType>
void writeValueMap(edm::Event &iEvent,
                   const edm::Handle<HandleType>& handle,
                   const std::vector<ValueType>& values,
                   const std::string& label)
{
    auto valMap = std::make_unique<edm::ValueMap<ValueType>>();
    typename edm::ValueMap<ValueType>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(std::move(valMap),label);
}

#endif
