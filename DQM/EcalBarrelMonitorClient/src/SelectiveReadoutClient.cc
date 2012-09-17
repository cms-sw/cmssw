#include "../interface/SelectiveReadoutClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <cmath>

namespace ecaldqm {

  SelectiveReadoutClient::SelectiveReadoutClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "SelectiveReadoutClient")
  {
  }

  void
  SelectiveReadoutClient::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    for(unsigned iME(0); iME < nMESets; ++iME)
      MEs_[iME]->resetAll(-1.);
  }

  void
  SelectiveReadoutClient::producePlots()
  {
    using namespace std;

    MEs_[kFRDropped]->reset();
    MEs_[kZSReadout]->reset();
    MEs_[kFR]->reset();
    MEs_[kRUForced]->reset();
    MEs_[kZS1]->reset();
    MEs_[kHighInterest]->reset();
    MEs_[kMedInterest]->reset();
    MEs_[kLowInterest]->reset();

    MESet::const_iterator ruItr(sources_[kRUForcedMap]);
    MESet::const_iterator frItr(sources_[kFullReadoutMap]);
    MESet::const_iterator zs1Itr(sources_[kZS1Map]);
    MESet::const_iterator zsItr(sources_[kZSMap]);
    MESet::const_iterator zsfrItr(sources_[kZSFullReadoutMap]);
    MESet::const_iterator frdItr(sources_[kFRDroppedMap]);

    MESet::iterator frdRateItr(MEs_[kFRDropped]);
    MESet::iterator zsrRateItr(MEs_[kZSReadout]);
    MESet::iterator frRateItr(MEs_[kFR]);
    MESet::iterator ruRateItr(MEs_[kRUForced]);
    MESet::iterator zs1RateItr(MEs_[kZS1]);

    MESet::const_iterator cEnd(sources_[kFlagCounterMap]->end());
    for(MESet::const_iterator cItr(sources_[kFlagCounterMap]->beginChannel()); cItr != cEnd; cItr.toNextChannel()){

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

      float nHigh(sources_[kHighIntMap]->getBinContent(id));
      float nMed(sources_[kMedIntMap]->getBinContent(id));
      float nLow(sources_[kLowIntMap]->getBinContent(id));
      float total(nHigh + nMed + nLow);

      if(total > 0.){
        MEs_[kHighInterest]->setBinContent(id, nHigh / total);
        MEs_[kMedInterest]->setBinContent(id, nMed / total);
        MEs_[kLowInterest]->setBinContent(id, nLow / total);
      }
    }
  }

  /*static*/
  void
  SelectiveReadoutClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["FRDropped"] = kFRDropped;
    _nameToIndex["ZSReadout"] = kZSReadout;
    _nameToIndex["FR"] = kFR;
    _nameToIndex["RUForced"] = kRUForced;
    _nameToIndex["ZS1"] = kZS1;
    _nameToIndex["HighInterest"] = kHighInterest;
    _nameToIndex["MedInterest"] = kMedInterest;
    _nameToIndex["LowInterest"] = kLowInterest;

    _nameToIndex["FlagCounterMap"] = kFlagCounterMap;
    _nameToIndex["RUForcedMap"] = kRUForcedMap;
    _nameToIndex["FullReadoutMap"] = kFullReadoutMap;
    _nameToIndex["ZS1Map"] = kZS1Map;
    _nameToIndex["ZSMap"] = kZSMap;
    _nameToIndex["ZSFullReadoutMap"] = kZSFullReadoutMap;
    _nameToIndex["FRDroppedMap"] = kFRDroppedMap;
    _nameToIndex["HighIntMap"] = kHighIntMap;
    _nameToIndex["MedIntMap"] = kMedIntMap;
    _nameToIndex["LowIntMap"] = kLowIntMap;
  }

  DEFINE_ECALDQM_WORKER(SelectiveReadoutClient);
}
