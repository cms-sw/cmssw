#include "../interface/SummaryClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  SummaryClient::SummaryClient(edm::ParameterSet const&  _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "SummaryClient"),
    towerBadFraction_(_workerParams.getUntrackedParameter<double>("towerBadFraction")),
    fedBadFraction_(_workerParams.getUntrackedParameter<double>("fedBadFraction"))
  {
    usedSources_.clear();
    usedSources_.insert("Integrity");
    usedSources_.insert("IntegrityByLumi");
    usedSources_.insert("RawData");
    usedSources_.insert("DesyncByLumi");
    usedSources_.insert("FEByLumi");

    std::vector<std::string> sourceList(_workerParams.getUntrackedParameter<std::vector<std::string> >("activeSources"));
    for(unsigned iS(0); iS < sourceList.size(); ++iS){
      std::string& sourceName(sourceList[iS]);
      if(sourceName == "Presample") usedSources_.insert("Presample");
      else if(sourceName == "Timing") usedSources_.insert("Timing");
      else if(sourceName == "TriggerPrimitives") usedSources_.insert("TriggerPrimitives");
      else if(sourceName == "HotCell") usedSources_.insert("HotCell");
    }

    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("ReportSummaryMap");
    qualitySummaries_.insert("ReportSummaryContents");
    qualitySummaries_.insert("ReportSummary");
  }

  void
  SummaryClient::bookMEs()
  {
    for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
      if(mItr->first == "NBadFEDs" && !online) continue;
      if(mItr->second){
        if(mItr->second->getBinType() == BinService::kTrend && !online) continue;
        if(!mItr->second->isActive()) mItr->second->book();
      }
    }
  }

  void
  SummaryClient::producePlots()
  {
    bool usePresample(using_("Presample"));
    bool useHotCell(using_("HotCell"));
    bool useTiming(using_("Timing"));
    bool useTrigPrim(using_("TriggerPrimitives"));

    MESet* meQualitySummary(MEs_["QualitySummary"]);
    MESet* meReportSummaryMap(MEs_["ReportSummaryMap"]);
    MESet* meReportSummaryContents(MEs_["ReportSummaryContents"]);
    MESet* meReportSummary(MEs_["ReportSummary"]);
    MESet* meNBadFEDs(online ? MEs_["NBadFEDs"] : 0);

    MESet const* sIntegrity(sources_["Integrity"]);
    MESet const* sIntegrityByLumi(sources_["IntegrityByLumi"]);
    MESet const* sPresample(usePresample ? sources_["Presample"] : 0);
    MESet const* sTiming(useTiming ? sources_["Timing"] : 0);
    MESet const* sRawData(sources_["RawData"]);
    MESet const* sDesyncByLumi(sources_["DesyncByLumi"]);
    MESet const* sFEByLumi(sources_["FEByLumi"]);
    MESet const* sTriggerPrimitives(useTrigPrim ? sources_["TriggerPrimitives"] : 0);
    MESet const* sHotCell(useHotCell ? sources_["HotCell"] : 0);

    float totalChannels(0.);
    float totalGood(0.);

    std::vector<float> dccChannels(BinService::nDCC, 0.);
    std::vector<float> dccGood(BinService::nDCC, 0.);

    std::vector<float> integrityByLumi(BinService::nDCC, 0.);
    std::vector<float> rawDataByLumi(BinService::nDCC, 0.);
    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      integrityByLumi[iDCC] = sIntegrityByLumi->getBinContent(iDCC + 1);
      rawDataByLumi[iDCC] = sDesyncByLumi->getBinContent(iDCC + 1) + sFEByLumi->getBinContent(iDCC + 1);
    }

    std::map<uint32_t, int> badChannelsCount;

    MESet::iterator qEnd(meQualitySummary->end());
    MESet::const_iterator iItr(sIntegrity);
    MESet::const_iterator pItr(sPresample, usePresample ? 0 : -1, 0);
    for(MESet::iterator qItr(meQualitySummary->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      DetId id(qItr->getId());
      unsigned iDCC(dccId(id) - 1);

      iItr = qItr;

      int integrity(iItr->getBinContent());

      if(integrity == kUnknown){
        qItr->setBinContent(integrity);
        continue;
      }

      pItr = qItr;

      int presample(usePresample ? pItr->getBinContent() : kUnknown);
      int hotcell(useHotCell ? sHotCell->getBinContent(id) : kUnknown);
      int timing(useTiming ? sTiming->getBinContent(id) : kUnknown);
      int trigprim(useTrigPrim ? sTriggerPrimitives->getBinContent(id) : kUnknown);

      int rawdata(sRawData->getBinContent(id));

      // summary retains only problems during this LS
      if(integrity == kBad && integrityByLumi[iDCC] == 0.) integrity = kGood;
      if(rawdata == kBad && rawDataByLumi[iDCC] == 0.) rawdata = kGood;

      int status(kGood);
      if(integrity == kBad || presample == kBad || timing == kBad || rawdata == kBad || trigprim == kBad || hotcell == kBad)
        status = kBad;
      else if(integrity == kUnknown && presample == kUnknown && timing == kUnknown && rawdata == kUnknown && trigprim == kUnknown)
        status = kUnknown;

      qItr->setBinContent(status);

      if(status == kBad){
        if(id.subdetId() == EcalBarrel) badChannelsCount[EBDetId(id).tower().rawId()] += 1;
        if(id.subdetId() == EcalEndcap) badChannelsCount[EEDetId(id).sc().rawId()] += 1;
      }
      else{
        dccGood[iDCC] += 1.;
        totalGood += 1.;
      }
      dccChannels[iDCC] += 1.;
      totalChannels += 1.;
    }

    // search clusters of bad towers
    if(online){
      for(int iz(-1); iz < 2; iz += 2){
        for(int ieta(0); ieta < 17; ++ieta){
          if(iz == 1 && ieta == 0) continue;
          for(int iphi(1); iphi <= 72; ++iphi){
            EcalTrigTowerDetId ttids[4];
            unsigned badTowers(0);
            for(int deta(0); deta < 2; ++deta){
              int ttz(ieta == 0 && deta == 0 ? -1 : iz);
              int tteta(ieta == 0 && deta == 0 ? 1 : ieta + deta);
              for(int dphi(0); dphi < 2; ++dphi){
                int ttphi(iphi != 72 ? iphi + dphi : 1);
                EcalTrigTowerDetId ttid(ttz, EcalBarrel, tteta, ttphi);
                ttids[deta * 2 + dphi] = ttid;

                if(badChannelsCount[ttid.rawId()] > towerBadFraction_ * 25.)
                  badTowers += 1;
              }
            }

            if(badTowers > 2){
              for(unsigned iD(0); iD < 4; ++iD)
                dccGood[dccId(ttids[iD]) - 1] = 0.;
            }
          }
        }
      }
      for(int iz(-1); iz <= 1; iz += 2){
        for(int ix(1); ix < 20; ++ix){
          for(int iy(1); iy < 20; ++iy){
            EcalScDetId scids[4];
            unsigned badTowers(0);
            for(int dx(0); dx < 2; ++dx){
              for(int dy(0); dy < 2; ++dy){
                if(!EcalScDetId::validDetId(ix + dx, iy + dy, iz)){
                  scids[dx * 2 + dy] = EcalScDetId(0);
                  continue;
                }
                EcalScDetId scid(ix + dx, iy + dy, iz);
                scids[dx * 2 + dy] = scid;

                if(badChannelsCount[scid.rawId()] > towerBadFraction_ * scConstituents(scid).size())
                  badTowers += 1;
              }
            }

            // contiguous towers bad -> [(00)(11)] [(11)(00)] [(01)(01)] [(10)(10)] []=>x ()=>y
            if(badTowers > 2){
              for(unsigned iD(0); iD < 4; ++iD){
                EcalScDetId& scid(scids[iD]);
                if(scid.null()) continue;
                dccGood[dccId(scid) - 1] = 0.;
              }
            }
          }
        }
      }
    }

    double nBad(0.);
    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      if(dccChannels[iDCC] < 1.) continue;

      unsigned dccid(iDCC + 1);
      float frac(dccGood[iDCC] / dccChannels[iDCC]);
      meReportSummaryMap->setBinContent(dccid, frac);
      meReportSummaryContents->fill(dccid, frac);

      if(1. - frac > fedBadFraction_) nBad += 1.;
    }

    if(totalChannels > 0.) meReportSummary->fill(totalGood / totalChannels);

    if(online) meNBadFEDs->setBinContent(1, nBad);
  }

  DEFINE_ECALDQM_WORKER(SummaryClient);
}



