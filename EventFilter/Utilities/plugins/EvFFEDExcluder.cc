#include <memory>
#include <algorithm>

#include "EvFFEDExcluder.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

evf::EvFFEDExcluder::EvFFEDExcluder(edm::ParameterSet const& config)
    : rawDataToken_(consumes(config.getParameter<edm::InputTag>("src"))),
      fedIds_([](std::vector<unsigned int> const& fedsToExclude) {
        // ret = all FED Ids except those to be excluded (i.e. fedsToExclude)
        std::vector<unsigned int> ret;
        auto const maxSize = FEDNumbering::lastFEDId() + 1;
        ret.reserve(maxSize);
        // loop on all FED IDs: [0, FEDNumbering::lastFEDId]
        for (auto fedId = 0u; fedId < maxSize; ++fedId) {
          // keep only fedIds not present in fedsToExclude
          if (std::find(fedsToExclude.begin(), fedsToExclude.end(), fedId) == fedsToExclude.end())
            ret.emplace_back(fedId);
        }
        ret.shrink_to_fit();
        return ret;
      }(config.getParameter<std::vector<unsigned int>>("fedsToExclude"))) {
  produces<FEDRawDataCollection>();
}

void evf::EvFFEDExcluder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("source"));
  desc.add<std::vector<unsigned int>>("fedsToExclude", {});
  descriptions.add("EvFFEDExcluder", desc);
}

void evf::EvFFEDExcluder::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  auto out = std::make_unique<FEDRawDataCollection>();
  auto const rawDataHandle = event.getHandle(rawDataToken_);

  for (auto const fedId : fedIds_)
    if (rawDataHandle->FEDData(fedId).size() > 0)
      out->FEDData(fedId) = rawDataHandle->FEDData(fedId);

  event.put(std::move(out));
}
