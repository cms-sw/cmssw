#include "DQMOffline/Lumi/interface/TriggerTools.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <string>

using namespace ZCountingTrigger;

//--------------------------------------------------------------------------------------------------
TriggerObjects TriggerTools::matchHLT(const double eta,
                                      const double phi,
                                      const std::vector<TriggerRecord>& triggerRecords,
                                      const trigger::TriggerEvent& triggerEvent) {
  const double dRMax = 0.2;

  TriggerObjects matchBits;
  for (const auto& triggerRecord : triggerRecords) {
    for (unsigned int iobj = 0; iobj < triggerRecord.objectMap.size(); iobj++) {
      const std::string& filterName = triggerRecord.objectMap[iobj].first;
      const unsigned int filterBit = triggerRecord.objectMap[iobj].second;

      edm::InputTag filterTag(filterName, "", "HLT");
      // filterIndex must be less than the size of trgEvent or you get a CMSException: _M_range_check
      if (triggerEvent.filterIndex(filterTag) < triggerEvent.sizeFilters()) {
        const trigger::TriggerObjectCollection& toc(triggerEvent.getObjects());
        const trigger::Keys& keys(triggerEvent.filterKeys(triggerEvent.filterIndex(filterTag)));

        for (unsigned short hltf : keys) {
          const trigger::TriggerObject& tobj(toc[hltf]);
          if (reco::deltaR(eta, phi, tobj.eta(), tobj.phi()) < dRMax) {
            matchBits[filterBit] = true;
          }
        }
      }
    }
  }

  return matchBits;
}
