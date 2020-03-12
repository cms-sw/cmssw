#include "DQM/EcalMonitorClient/interface/SummaryClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>

namespace ecaldqm {
  SummaryClient::SummaryClient() : DQWorkerClient(), towerBadFraction_(0.), fedBadFraction_(0.) {
    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("ReportSummaryMap");
    qualitySummaries_.insert("ReportSummaryContents");
    qualitySummaries_.insert("ReportSummary");
  }

  void SummaryClient::setParams(edm::ParameterSet const& _params) {
    towerBadFraction_ = _params.getUntrackedParameter<double>("towerBadFraction");
    fedBadFraction_ = _params.getUntrackedParameter<double>("fedBadFraction");

    std::vector<std::string> sourceList(_params.getUntrackedParameter<std::vector<std::string> >("activeSources"));
    if (std::find(sourceList.begin(), sourceList.end(), "Presample") == sourceList.end())
      sources_.erase(std::string("Presample"));
    if (std::find(sourceList.begin(), sourceList.end(), "Timing") == sourceList.end())
      sources_.erase(std::string("Timing"));
    if (std::find(sourceList.begin(), sourceList.end(), "TriggerPrimitives") == sourceList.end())
      sources_.erase(std::string("TriggerPrimitives"));
    if (std::find(sourceList.begin(), sourceList.end(), "HotCell") == sourceList.end())
      sources_.erase(std::string("HotCell"));
  }

  void SummaryClient::resetMEs() {
    DQWorkerClient::resetMEs();

    MESet& meReportSummaryContents(MEs_.at("ReportSummaryContents"));
    MESet& meReportSummary(MEs_.at("ReportSummary"));
    MESet& meReportSummaryMap(MEs_.at("ReportSummaryMap"));

    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
      int dccid(iDCC + 1);
      meReportSummaryContents.fill(dccid, -1.);
    }

    meReportSummary.fill(-1.);

    meReportSummaryMap.reset(-1.);
  }

  void SummaryClient::producePlots(ProcessType _pType) {
    MESet& meReportSummaryContents(MEs_.at("ReportSummaryContents"));
    MESet& meReportSummary(MEs_.at("ReportSummary"));

    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
      int dccid(iDCC + 1);
      meReportSummaryContents.fill(dccid, -1.);
    }
    meReportSummary.fill(-1.);

    MESet const& sIntegrityByLumi(sources_.at("IntegrityByLumi"));
    MESet const& sDesyncByLumi(sources_.at("DesyncByLumi"));
    MESet const& sFEByLumi(sources_.at("FEByLumi"));                          // Does NOT include FE=Disabled
    MESet const& sFEStatusErrMapByLumi(sources_.at("FEStatusErrMapByLumi"));  // Includes FE=Disabled

    double integrityByLumi[nDCC];
    double rawDataByLumi[nDCC];
    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
      integrityByLumi[iDCC] = sIntegrityByLumi.getBinContent(iDCC + 1);
      rawDataByLumi[iDCC] = sDesyncByLumi.getBinContent(iDCC + 1) + sFEByLumi.getBinContent(iDCC + 1);
    }

    MESet& meQualitySummary(MEs_.at("QualitySummary"));
    MESet& meReportSummaryMap(MEs_.at("ReportSummaryMap"));

    MESet const* sIntegrity(using_("Integrity") ? &sources_.at("Integrity") : nullptr);
    MESet const& sRawData(sources_.at("RawData"));
    MESet const* sPresample(using_("Presample") ? &sources_.at("Presample") : nullptr);
    MESet const* sTiming(using_("Timing") ? &sources_.at("Timing") : nullptr);
    MESet const* sTriggerPrimitives(using_("TriggerPrimitives") ? &sources_.at("TriggerPrimitives") : nullptr);
    MESet const* sHotCell(using_("HotCell") ? &sources_.at("HotCell") : nullptr);

    float totalChannels(0.);
    float totalGood(0.), totalGoodRaw(0);

    double dccChannels[nDCC];
    std::fill_n(dccChannels, nDCC, 0.);
    double dccGood[nDCC], dccGoodRaw[nDCC];
    std::fill_n(dccGood, nDCC, 0.);
    std::fill_n(dccGoodRaw, nDCC, 0.);

    std::map<uint32_t, int> badChannelsCount;

    // Override IntegrityByLumi check if any Desync errors present
    // Used to set an entire FED to BAD
    MESet const& sBXSRP(sources_.at("BXSRP"));
    MESet const& sBXTCC(sources_.at("BXTCC"));
    std::vector<bool> hasMismatchDCC(nDCC, false);
    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
      if (sBXSRP.getBinContent(iDCC + 1) > 50. || sBXTCC.getBinContent(iDCC + 1) > 50.)  // "any" = 50
        hasMismatchDCC[iDCC] = true;
    }

    // Get RawData mask
    uint32_t mask(1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR);

    MESet::iterator qEnd(meQualitySummary.end());
    for (MESet::iterator qItr(meQualitySummary.beginChannel()); qItr != qEnd; qItr.toNextChannel()) {
      DetId id(qItr->getId());
      unsigned iDCC(dccId(id) - 1);

      // Initialize individual Quality Summaries
      // NOTE: These represent quality over *cumulative* statistics
      int integrity(sIntegrity ? (int)sIntegrity->getBinContent(id) : kUnknown);
      int presample(sPresample ? (int)sPresample->getBinContent(id) : kUnknown);
      int hotcell(sHotCell ? (int)sHotCell->getBinContent(id) : kUnknown);
      int timing(sTiming ? (int)sTiming->getBinContent(id) : kUnknown);
      int trigprim(sTriggerPrimitives ? (int)sTriggerPrimitives->getBinContent(id) : kUnknown);
      int rawdata(sRawData.getBinContent(id));

      double rawdataLS(sFEStatusErrMapByLumi.getBinContent(id));  // Includes FE=Disabled

      // If there are no RawData or Integrity errors in this LS, set them back to GOOD
      //if(integrity == kBad && integrityByLumi[iDCC] == 0.) integrity = kGood;
      if (integrity == kBad && integrityByLumi[iDCC] == 0. && !hasMismatchDCC[iDCC])
        integrity = kGood;
      //if(rawdata == kBad && rawDataByLumi[iDCC] == 0.) rawdata = kGood;
      if (rawdata == kBad && rawDataByLumi[iDCC] == 0. && rawdataLS == 0.)
        rawdata = kGood;

      // Fill Global Quality Summary
      int status(kGood);
      if (integrity == kBad || presample == kBad || timing == kBad || rawdata == kBad || trigprim == kBad ||
          hotcell == kBad)
        status = kBad;
      else if (integrity == kUnknown && presample == kUnknown && timing == kUnknown && rawdata == kUnknown &&
               trigprim == kUnknown)
        status = kUnknown;
      // Skip channels with no/low integrity statistics (based on digi occupancy)
      // Normally, ensures Global Quality and Report Summaries are not filled when stats are still low / channel masked / ECAL not in run
      // However, problematic FEDs can sometimes drop hits so check that channel is not flagged as BAD elsewhere
      if (status != kBad && (integrity == kUnknown || integrity == kMUnknown)) {
        qItr->setBinContent(integrity);
        if (onlineMode_)
          continue;
      }
      qItr->setBinContent(status);

      // Keep running count of good/bad channels/towers: Uses cumulative stats.
      if (status == kBad) {
        if (id.subdetId() == EcalBarrel)
          badChannelsCount[EBDetId(id).tower().rawId()] += 1;
        if (id.subdetId() == EcalEndcap)
          badChannelsCount[EEDetId(id).sc().rawId()] += 1;
      } else {
        dccGood[iDCC] += 1.;
        totalGood += 1.;
      }
      dccChannels[iDCC] += 1.;
      totalChannels += 1.;

      // Keep running count of good channels in RawData only: Uses LS stats only.
      // LS-based reports only use RawData as input to save on having to run other workers
      bool isMasked(meQualitySummary.maskMatches(id, mask, statusManager_));
      if (rawdataLS == 0. || isMasked) {  // channel != kBad in rawdata
        dccGoodRaw[iDCC] += 1.;
        totalGoodRaw += 1.;
      }

    }  // qItr channel loop

    // search clusters of bad towers
    /*if(onlineMode_){

      // EB
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
              } // dphi
            } // deta
            if(badTowers > 2){
              for(unsigned iD(0); iD < 4; ++iD)
                dccGood[dccId(ttids[iD]) - 1] = 0.;
            }
          } // iphi
        } // ieta
      } // iz

      // EE
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
              } // dy
            } // dx
            // contiguous towers bad -> [(00)(11)] [(11)(00)] [(01)(01)] [(10)(10)] []=>x ()=>y
            if(badTowers > 2){
              for(unsigned iD(0); iD < 4; ++iD){
                EcalScDetId& scid(scids[iD]);
                if(scid.null()) continue;
                dccGood[dccId(scid) - 1] = 0.;
              }
            }
          } // iy
        } // ix
      } // iz

    } // cluster search */

    // Fill Report Summaries
    double nBad(0.);
    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
      if (dccChannels[iDCC] < 1.)
        continue;

      int dccid(iDCC + 1);
      float frac(dccGood[iDCC] / dccChannels[iDCC]);
      float fracRaw(dccGoodRaw[iDCC] / dccChannels[iDCC]);
      meReportSummaryMap.setBinContent(dccid, frac);
      float fracLS(onlineMode_ ? frac : fracRaw);
      meReportSummaryContents.fill(dccid, fracLS);  // reported by LS

      if (1. - frac > fedBadFraction_)
        nBad += 1.;
    }

    float totalGoodLS(onlineMode_ ? totalGood : totalGoodRaw);
    if (totalChannels > 0.)
      meReportSummary.fill(totalGoodLS / totalChannels);  // reported by LS

    if (onlineMode_) {
      if (totalChannels > 0.)
        MEs_.at("GlobalSummary").setBinContent(1, totalGood / totalChannels);
      MEs_.at("NBadFEDs").setBinContent(1, nBad);
    }

  }  // producePlots()

  DEFINE_ECALDQM_WORKER(SummaryClient);
}  // namespace ecaldqm
