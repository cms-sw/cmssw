#include "../interface/SummaryClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  SummaryClient::SummaryClient(edm::ParameterSet const&  _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "SummaryClient"),
    towerBadFraction_(_workerParams.getUntrackedParameter<double>("towerBadFraction")),
    fedBadFraction_(_workerParams.getUntrackedParameter<double>("fedBadFraction"))
  {
    usedSources_.clear();
    use_(kIntegrity);
    use_(kIntegrityByLumi);
    use_(kRawData);
    use_(kDesyncByLumi);
    use_(kFEByLumi);

    std::vector<std::string> sourceList(_workerParams.getUntrackedParameter<std::vector<std::string> >("activeSources"));
    for(unsigned iS(0); iS < sourceList.size(); ++iS){
      std::string& sourceName(sourceList[iS]);
      if(sourceName == "Presample") use_(kPresample);
      else if(sourceName == "Timing") use_(kTiming);
      else if(sourceName == "TriggerPrimitives") use_(kTriggerPrimitives);
      else if(sourceName == "HotCell") use_(kHotCell);
    }

    qualitySummaries_.insert(kQualitySummary);
    qualitySummaries_.insert(kReportSummaryMap);
    qualitySummaries_.insert(kReportSummaryContents);
    qualitySummaries_.insert(kReportSummary);
  }

  void
  SummaryClient::bookMEs()
  {
    for(unsigned iME(0); iME < MEs_.size(); iME++){
      if(iME == kNBadFEDs && !online) continue;
      if(MEs_[iME]){
        if(MEs_[iME]->getBinType() == BinService::kTrend && !online) continue;
        if(!MEs_[iME]->isActive()) MEs_[iME]->book();
      }
    }
  }

  void
  SummaryClient::producePlots()
  {
    float totalChannels(0.);
    float totalGood(0.);

    std::vector<float> dccChannels(BinService::nDCC, 0.);
    std::vector<float> dccGood(BinService::nDCC, 0.);

    std::vector<float> integrityByLumi(BinService::nDCC, 0.);
    std::vector<float> rawDataByLumi(BinService::nDCC, 0.);
    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      integrityByLumi[iDCC] = sources_[kIntegrityByLumi]->getBinContent(iDCC + 1);
      rawDataByLumi[iDCC] = sources_[kDesyncByLumi]->getBinContent(iDCC + 1) + sources_[kFEByLumi]->getBinContent(iDCC + 1);
    }

    std::map<uint32_t, int> badChannelsCount;

    MESet::iterator qEnd(MEs_[kQualitySummary]->end());
    MESet::const_iterator iItr(sources_[kIntegrity]);
    MESet::const_iterator pItr(sources_[kPresample], using_(kPresample) ? 0 : -1, 0);
    for(MESet::iterator qItr(MEs_[kQualitySummary]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      DetId id(qItr->getId());
      unsigned iDCC(dccId(id) - 1);

      iItr = qItr;

      int integrity(iItr->getBinContent());

      if(integrity == kUnknown){
        qItr->setBinContent(integrity);
        continue;
      }

      pItr = qItr;

      int presample(using_(kPresample) ? pItr->getBinContent() : kUnknown);
      int hotcell(using_(kHotCell) ? sources_[kHotCell]->getBinContent(id) : kUnknown);
      int timing(using_(kTiming) ? sources_[kTiming]->getBinContent(id) : kUnknown);
      int trigprim(using_(kTriggerPrimitives) ? sources_[kTriggerPrimitives]->getBinContent(id) : kUnknown);

      int rawdata(sources_[kRawData]->getBinContent(id));

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
      MEs_[kReportSummaryMap]->setBinContent(dccid, frac);
      MEs_[kReportSummaryContents]->fill(dccid, frac);

      if(1. - frac > fedBadFraction_) nBad += 1.;
    }

    if(totalChannels > 0.) MEs_[kReportSummary]->fill(totalGood / totalChannels);

    if(online) MEs_[kNBadFEDs]->setBinContent(1, nBad);
  }

  /*static*/
  void
  SummaryClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["QualitySummary"] = kQualitySummary;
    _nameToIndex["ReportSummaryMap"] = kReportSummaryMap;
    _nameToIndex["ReportSummaryContents"] = kReportSummaryContents;
    _nameToIndex["ReportSummary"] = kReportSummary;
    _nameToIndex["NBadFEDs"] = kNBadFEDs;

    _nameToIndex["Integrity"] = kIntegrity;
    _nameToIndex["IntegrityByLumi"] = kIntegrityByLumi;
    _nameToIndex["Presample"] = kPresample;
    _nameToIndex["Timing"] = kTiming;
    _nameToIndex["RawData"] = kRawData;
    _nameToIndex["DesyncByLumi"] = kDesyncByLumi;
    _nameToIndex["FEByLumi"] = kFEByLumi;
    _nameToIndex["TriggerPrimitives"] = kTriggerPrimitives;
    _nameToIndex["HotCell"] = kHotCell;
  }

  DEFINE_ECALDQM_WORKER(SummaryClient);
}

