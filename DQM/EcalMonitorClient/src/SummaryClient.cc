#include "../interface/SummaryClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>

namespace ecaldqm
{
  SummaryClient::SummaryClient() :
    DQWorkerClient(),
    towerBadFraction_(0.),
    fedBadFraction_(0.)
  {
    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("ReportSummaryMap");
    qualitySummaries_.insert("ReportSummaryContents");
    qualitySummaries_.insert("ReportSummary");
  }

  void
  SummaryClient::setParams(edm::ParameterSet const& _params)
  {
    towerBadFraction_ = _params.getUntrackedParameter<double>("towerBadFraction");
    fedBadFraction_ = _params.getUntrackedParameter<double>("fedBadFraction");

    std::vector<std::string> sourceList(_params.getUntrackedParameter<std::vector<std::string> >("activeSources"));
    if(std::find(sourceList.begin(), sourceList.end(), "Presample") == sourceList.end()) sources_.erase(std::string("Presample"));
    if(std::find(sourceList.begin(), sourceList.end(), "Timing") == sourceList.end()) sources_.erase(std::string("Timing"));
    if(std::find(sourceList.begin(), sourceList.end(), "TriggerPrimitives") == sourceList.end()) sources_.erase(std::string("TriggerPrimitives"));
    if(std::find(sourceList.begin(), sourceList.end(), "HotCell") == sourceList.end()) sources_.erase(std::string("HotCell"));
  }

  void
  SummaryClient::resetMEs()
  {
    DQWorkerClient::resetMEs();

    MESet& meReportSummaryContents(MEs_.at("ReportSummaryContents"));
    MESet& meReportSummary(MEs_.at("ReportSummary"));
    MESet& meReportSummaryMap(MEs_.at("ReportSummaryMap"));

    for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){
      int dccid(iDCC + 1);
      meReportSummaryContents.fill(dccid, -1.);
    }

    meReportSummary.fill(-1.);

    meReportSummaryMap.reset(-1.);
  }

  void
  SummaryClient::producePlots(ProcessType _pType)
  {
    if(_pType == kLumi && !onlineMode_) return;
    // TODO: Implement offline per-lumi summary

    MESet& meReportSummaryContents(MEs_.at("ReportSummaryContents"));
    MESet& meReportSummary(MEs_.at("ReportSummary"));

    for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){
      int dccid(iDCC + 1);
      meReportSummaryContents.fill(dccid, -1.);
    }
    meReportSummary.fill(-1.);

    MESet const& sIntegrityByLumi(sources_.at("IntegrityByLumi"));
    MESet const& sDesyncByLumi(sources_.at("DesyncByLumi"));
    MESet const& sFEByLumi(sources_.at("FEByLumi"));

    double integrityByLumi[nDCC];
    double rawDataByLumi[nDCC];
    for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){
      integrityByLumi[iDCC] = sIntegrityByLumi.getBinContent(iDCC + 1);
      rawDataByLumi[iDCC] = sDesyncByLumi.getBinContent(iDCC + 1) + sFEByLumi.getBinContent(iDCC + 1);
    }

    MESet& meQualitySummary(MEs_.at("QualitySummary"));
    MESet& meReportSummaryMap(MEs_.at("ReportSummaryMap"));

    MESet const& sIntegrity(sources_.at("Integrity"));
    MESet const& sRawData(sources_.at("RawData"));
    MESet const* sPresample(using_("Presample") ? &sources_.at("Presample") : 0);
    MESet const* sTiming(using_("Timing") ? &sources_.at("Timing") : 0);
    MESet const* sTriggerPrimitives(using_("TriggerPrimitives") ? &sources_.at("TriggerPrimitives") : 0);
    MESet const* sHotCell(using_("HotCell") ? &sources_.at("HotCell") : 0);

    float totalChannels(0.);
    float totalGood(0.);

    double dccChannels[nDCC];
    std::fill_n(dccChannels, nDCC, 0.);
    double dccGood[nDCC];
    std::fill_n(dccGood, nDCC, 0.);

    std::map<uint32_t, int> badChannelsCount;

    MESet::iterator qEnd(meQualitySummary.end());
    MESet::const_iterator iItr(sIntegrity);
    for(MESet::iterator qItr(meQualitySummary.beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      DetId id(qItr->getId());
      unsigned iDCC(dccId(id) - 1);

      iItr = qItr;

      int integrity(iItr->getBinContent());

      if(integrity == kUnknown || integrity == kMUnknown){
        qItr->setBinContent(integrity);
        continue;
      }

      int presample(sPresample ? sPresample->getBinContent(id) : kUnknown);
      int hotcell(sHotCell ? sHotCell->getBinContent(id) : kUnknown);
      int timing(sTiming ? sTiming->getBinContent(id) : kUnknown);
      int trigprim(sTriggerPrimitives ? sTriggerPrimitives->getBinContent(id) : kUnknown);

      int rawdata(sRawData.getBinContent(id));

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
    if(onlineMode_){
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
    for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){
      if(dccChannels[iDCC] < 1.) continue;

      int dccid(iDCC + 1);
      float frac(dccGood[iDCC] / dccChannels[iDCC]);
      meReportSummaryMap.setBinContent(dccid, frac);
      meReportSummaryContents.fill(dccid, frac);

      if(1. - frac > fedBadFraction_) nBad += 1.;
    }

    if(totalChannels > 0.) meReportSummary.fill(totalGood / totalChannels);

    if(onlineMode_){
      if(totalChannels > 0.) MEs_.at("GlobalSummary").setBinContent(1, totalGood / totalChannels);
      MEs_.at("NBadFEDs").setBinContent(1, nBad);
    }
  }

  DEFINE_ECALDQM_WORKER(SummaryClient);
}



