#include "../interface/SelectiveReadoutClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <cmath>

namespace ecaldqm {

  SelectiveReadoutClient::SelectiveReadoutClient(const edm::ParameterSet& _params) :
    DQWorkerClient(_params, "SelectiveReadoutClient")
  {
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

    for(unsigned dccid(1); dccid <= 54; dccid++){

      for(unsigned tower(1); tower <= nSuperCrystals(dccid); tower++){
	vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	if(ids.size() == 0) continue;

	float nFlags(0.);
	float nRUForced(0.);
	float nFullReadoutFlags(0.);
	float nZS1Flags(0.);
	float nZS12Flags(0.);
	float nZSFullReadout(0.);
	float nFRDropped(0.);
	if(dccid <= 9 || dccid >= 46){
	  vector<EcalScDetId> scids(getElectronicsMap()->getEcalScDetId(dccid, tower));
	  for(vector<EcalScDetId>::iterator scItr(scids.begin()); scItr != scids.end(); ++scItr){
	    nFlags += sources_[sFlagCounterMap]->getBinContent(*scItr);
	    nRUForced += sources_[sRUForcedMap]->getBinContent(*scItr);
	    nFullReadoutFlags += sources_[sFullReadoutMap]->getBinContent(*scItr);
	    nZS1Flags += sources_[sZS1Map]->getBinContent(*scItr);
	    nZS12Flags += sources_[sZSMap]->getBinContent(*scItr);
	    nZSFullReadout += sources_[sZSFullReadoutMap]->getBinContent(*scItr);
	    nFRDropped += sources_[sFRDroppedMap]->getBinContent(*scItr);
	  }

	  for(vector<EcalScDetId>::iterator scItr(scids.begin()); scItr != scids.end(); ++scItr){
	    if(nFlags > 0.) MEs_[kFR]->setBinContent(*scItr, nFullReadoutFlags / nFlags);
	    if(nFlags > 0.) MEs_[kZS1]->setBinContent(*scItr, nZS1Flags / nFlags);
	    if(nFlags > 0.) MEs_[kRUForced]->setBinContent(*scItr, nRUForced / nFlags);
	    if(nZS12Flags > 0.) MEs_[kZSReadout]->setBinContent(*scItr, nZSFullReadout / nZS12Flags);
	    if(nFullReadoutFlags > 0.) MEs_[kFRDropped]->setBinContent(*scItr, nFRDropped / nFullReadoutFlags);
	  }
	}
	else{
	  nFlags = sources_[sFlagCounterMap]->getBinContent(ids[0]);
	  nRUForced = sources_[sRUForcedMap]->getBinContent(ids[0]);
	  nFullReadoutFlags = sources_[sFullReadoutMap]->getBinContent(ids[0]);
	  nZS1Flags = sources_[sZS1Map]->getBinContent(ids[0]);
	  nZS12Flags = sources_[sZSMap]->getBinContent(ids[0]);
	  nZSFullReadout = sources_[sZSFullReadoutMap]->getBinContent(ids[0]);
	  nFRDropped = sources_[sFRDroppedMap]->getBinContent(ids[0]);


	  if(nFlags > 0.) MEs_[kFR]->setBinContent(ids[0], nFullReadoutFlags / nFlags);
	  if(nFlags > 0.) MEs_[kZS1]->setBinContent(ids[0], nZS1Flags / nFlags);
	  if(nFlags > 0.) MEs_[kRUForced]->setBinContent(ids[0], nRUForced / nFlags);
	  if(nZS12Flags > 0.) MEs_[kZSReadout]->setBinContent(ids[0], nZSFullReadout / nZS12Flags);
	  if(nFullReadoutFlags > 0.) MEs_[kFRDropped]->setBinContent(ids[0], nFRDropped / nFullReadoutFlags);
	}
      }
    }
  }

  /*static*/
  void
  SelectiveReadoutClient::setMEData(std::vector<MEData>& _data)
  {
    _data[kFRDropped] = MEData("FRDropped", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kZSReadout] = MEData("ZSReadout", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kFR] = MEData("FR", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kRUForced] = MEData("RUForced", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kZS1] = MEData("ZS1", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);

    _data[sFlagCounterMap + nTargets] = MEData("FlagCounterMap");
    _data[sRUForcedMap + nTargets] = MEData("RUForcedMap");
    _data[sFullReadoutMap + nTargets] = MEData("FullReadoutMap");
    _data[sZS1Map + nTargets] = MEData("ZS1Map");
    _data[sZSMap + nTargets] = MEData("ZSMap");
    _data[sZSFullReadoutMap + nTargets] = MEData("ZSFullReadoutMap");
    _data[sFRDroppedMap + nTargets] = MEData("FRDroppedMap");
  }

  DEFINE_ECALDQM_WORKER(SelectiveReadoutClient);
}

