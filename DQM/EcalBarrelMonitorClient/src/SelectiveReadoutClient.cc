#include "../interface/SelectiveReadoutClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <cmath>

namespace ecaldqm {

  SelectiveReadoutClient::SelectiveReadoutClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "SelectiveReadoutClient")
  {
  }

  void
  SelectiveReadoutClient::producePlots()
  {
    MESet* meFRDropped(MEs_["FRDropped"]);
    MESet* meZSReadout(MEs_["ZSReadout"]);
    MESet* meFR(MEs_["FR"]);
    MESet* meRUForced(MEs_["RUForced"]);
    MESet* meZS1(MEs_["ZS1"]);
    MESet* meHighInterest(MEs_["HighInterest"]);
    MESet* meMedInterest(MEs_["MedInterest"]);
    MESet* meLowInterest(MEs_["LowInterest"]);

    MESet const* sFlagCounterMap(sources_["FlagCounterMap"]);
    MESet const* sRUForcedMap(sources_["RUForcedMap"]);
    MESet const* sFullReadoutMap(sources_["FullReadoutMap"]);
    MESet const* sZS1Map(sources_["ZS1Map"]);
    MESet const* sZSMap(sources_["ZSMap"]);
    MESet const* sZSFullReadoutMap(sources_["ZSFullReadoutMap"]);
    MESet const* sFRDroppedMap(sources_["FRDroppedMap"]);
    MESet const* sHighIntMap(sources_["HighIntMap"]);
    MESet const* sMedIntMap(sources_["MedIntMap"]);
    MESet const* sLowIntMap(sources_["LowIntMap"]);

    meFRDropped->reset();
    meZSReadout->reset();
    meFR->reset();
    meRUForced->reset();
    meZS1->reset();
    meHighInterest->reset();
    meMedInterest->reset();
    meLowInterest->reset();

    MESet::const_iterator ruItr(sRUForcedMap);
    MESet::const_iterator frItr(sFullReadoutMap);
    MESet::const_iterator zs1Itr(sZS1Map);
    MESet::const_iterator zsItr(sZSMap);
    MESet::const_iterator zsfrItr(sZSFullReadoutMap);
    MESet::const_iterator frdItr(sFRDroppedMap);

    MESet::iterator frdRateItr(meFRDropped);
    MESet::iterator zsrRateItr(meZSReadout);
    MESet::iterator frRateItr(meFR);
    MESet::iterator ruRateItr(meRUForced);
    MESet::iterator zs1RateItr(meZS1);

    MESet::const_iterator cEnd(sFlagCounterMap->end());
    for(MESet::const_iterator cItr(sFlagCounterMap->beginChannel()); cItr != cEnd; cItr.toNextChannel()){

      ruItr = cItr;
      frItr = cItr;
      zs1Itr = cItr;
      zsItr = cItr;
      zsfrItr = cItr;
      frdItr = cItr;

      frdRateItr = cItr;
      zsrRateItr = cItr;
      frRateItr = cItr;
      ruRateItr = cItr;
      zs1RateItr = cItr;

      float nFlags(cItr->getBinContent());
      float nZS12Flags(zsItr->getBinContent());
      float nFullReadoutFlags(frItr->getBinContent());

      if(nFlags > 0.){
        frRateItr->setBinContent(nFullReadoutFlags / nFlags);
        zs1RateItr->setBinContent(zs1Itr->getBinContent() / nFlags);
        ruRateItr->setBinContent(ruItr->getBinContent() / nFlags);
      }
      if(nZS12Flags > 0.)
        zsrRateItr->setBinContent(zsfrItr->getBinContent() / nZS12Flags);
      if(nFullReadoutFlags > 0.)
        frdRateItr->setBinContent(frdItr->getBinContent() / nFullReadoutFlags);

    }

    // iterator not supported for kTriggerTower binning yet
    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT){
      EcalTrigTowerDetId id(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));

      float nHigh(sHighIntMap->getBinContent(id));
      float nMed(sMedIntMap->getBinContent(id));
      float nLow(sLowIntMap->getBinContent(id));
      float total(nHigh + nMed + nLow);

      if(total > 0.){
        meHighInterest->setBinContent(id, nHigh / total);
        meMedInterest->setBinContent(id, nMed / total);
        meLowInterest->setBinContent(id, nLow / total);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(SelectiveReadoutClient);
}
