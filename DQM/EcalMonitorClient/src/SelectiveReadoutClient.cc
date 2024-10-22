#include "DQM/EcalMonitorClient/interface/SelectiveReadoutClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <cmath>

namespace ecaldqm {
  SelectiveReadoutClient::SelectiveReadoutClient() : DQWorkerClient() {}

  void SelectiveReadoutClient::producePlots(ProcessType) {
    MESet& meFRDropped(MEs_.at("FRDropped"));
    MESet& meZSReadout(MEs_.at("ZSReadout"));
    MESet& meFR(MEs_.at("FR"));
    MESet& meRUForced(MEs_.at("RUForced"));
    MESet& meZS1(MEs_.at("ZS1"));
    MESet& meHighInterest(MEs_.at("HighInterest"));
    MESet& meMedInterest(MEs_.at("MedInterest"));
    MESet& meLowInterest(MEs_.at("LowInterest"));

    MESet const& sFlagCounterMap(sources_.at("FlagCounterMap"));
    MESet const& sRUForcedMap(sources_.at("RUForcedMap"));
    MESet const& sFullReadoutMap(sources_.at("FullReadoutMap"));
    MESet const& sZS1Map(sources_.at("ZS1Map"));
    MESet const& sZSMap(sources_.at("ZSMap"));
    MESet const& sZSFullReadoutMap(sources_.at("ZSFullReadoutMap"));
    MESet const& sFRDroppedMap(sources_.at("FRDroppedMap"));
    MESet const& sHighIntMap(sources_.at("HighIntMap"));
    MESet const& sMedIntMap(sources_.at("MedIntMap"));
    MESet const& sLowIntMap(sources_.at("LowIntMap"));

    MESet::const_iterator ruItr(GetElectronicsMap(), sRUForcedMap);
    MESet::const_iterator frItr(GetElectronicsMap(), sFullReadoutMap);
    MESet::const_iterator zs1Itr(GetElectronicsMap(), sZS1Map);
    MESet::const_iterator zsItr(GetElectronicsMap(), sZSMap);
    MESet::const_iterator zsfrItr(GetElectronicsMap(), sZSFullReadoutMap);
    MESet::const_iterator frdItr(GetElectronicsMap(), sFRDroppedMap);

    MESet::iterator frdRateItr(GetElectronicsMap(), meFRDropped);
    MESet::iterator zsrRateItr(GetElectronicsMap(), meZSReadout);
    MESet::iterator frRateItr(GetElectronicsMap(), meFR);
    MESet::iterator ruRateItr(GetElectronicsMap(), meRUForced);
    MESet::iterator zs1RateItr(GetElectronicsMap(), meZS1);

    MESet::const_iterator cEnd(sFlagCounterMap.end(GetElectronicsMap()));
    for (MESet::const_iterator cItr(sFlagCounterMap.beginChannel(GetElectronicsMap())); cItr != cEnd;
         cItr.toNextChannel(GetElectronicsMap())) {
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

      if (nFlags > 0.) {
        frRateItr->setBinContent(nFullReadoutFlags / nFlags);
        zs1RateItr->setBinContent(zs1Itr->getBinContent() / nFlags);
        ruRateItr->setBinContent(ruItr->getBinContent() / nFlags);
      }
      if (nZS12Flags > 0.)
        zsrRateItr->setBinContent(zsfrItr->getBinContent() / nZS12Flags);
      if (nFullReadoutFlags > 0.)
        frdRateItr->setBinContent(frdItr->getBinContent() / nFullReadoutFlags);
    }

    // iterator not supported for kTriggerTower binning yet
    for (unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT) {
      EcalTrigTowerDetId id(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));

      float nHigh(sHighIntMap.getBinContent(getEcalDQMSetupObjects(), id));
      float nMed(sMedIntMap.getBinContent(getEcalDQMSetupObjects(), id));
      float nLow(sLowIntMap.getBinContent(getEcalDQMSetupObjects(), id));
      float total(nHigh + nMed + nLow);

      if (total > 0.) {
        meHighInterest.setBinContent(getEcalDQMSetupObjects(), id, nHigh / total);
        meMedInterest.setBinContent(getEcalDQMSetupObjects(), id, nMed / total);
        meLowInterest.setBinContent(getEcalDQMSetupObjects(), id, nLow / total);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(SelectiveReadoutClient);
}  // namespace ecaldqm
