#include "DQM/EcalMonitorClient/interface/IntegrityClient.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetNonObject.h"

namespace ecaldqm {
  IntegrityClient::IntegrityClient() : DQWorkerClient(), errFractionThreshold_(0.), processedEvents(0) {
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void IntegrityClient::setParams(edm::ParameterSet const& _params) {
    errFractionThreshold_ = _params.getUntrackedParameter<double>("errFractionThreshold");
  }

  void IntegrityClient::setTokens(edm::ConsumesCollector& _collector) {
    chStatusToken = _collector.esConsumes<edm::Transition::EndLuminosityBlock>();
  }

  // Check Channel Status Record at every endLumi
  // Used to fill Channel Status Map MEs
  void IntegrityClient::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& _es) {
    chStatus = &_es.getData(chStatusToken);
  }

  void IntegrityClient::producePlots(ProcessType) {
    uint32_t mask(1 << EcalDQMStatusHelper::CH_ID_ERROR | 1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR |
                  1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR | 1 << EcalDQMStatusHelper::TT_ID_ERROR |
                  1 << EcalDQMStatusHelper::TT_SIZE_ERROR);

    MESet& meQuality(MEs_.at("Quality"));
    MESet& meQualitySummary(MEs_.at("QualitySummary"));
    MESet& meChStatus(MEs_.at("ChStatus"));

    MESet const& sOccupancy(sources_.at("Occupancy"));
    MESet const& sGain(sources_.at("Gain"));
    MESet const& sChId(sources_.at("ChId"));
    MESet const& sGainSwitch(sources_.at("GainSwitch"));
    MESet const& sTowerId(sources_.at("TowerId"));
    MESet const& sBlockSize(sources_.at("BlockSize"));
    MESetNonObject const& sNumEvents(static_cast<MESetNonObject&>(sources_.at("NumEvents")));

    //Get the no.of events per LS calculated in OccupancyTask
    int nEv = sNumEvents.getFloatValue();
    processedEvents += nEv;  //Sum it up to get the total processed events for the whole run.

    //TTID errors nomalized by total no.of events in a run.
    MESet& meTTIDNorm(MEs_.at("TowerIdNormalized"));
    MESet::iterator tEnd(meTTIDNorm.end(GetElectronicsMap()));
    for (MESet::iterator tItr(meTTIDNorm.beginChannel(GetElectronicsMap())); tItr != tEnd;
         tItr.toNextChannel(GetElectronicsMap())) {
      DetId id(tItr->getId());
      float towerid(sTowerId.getBinContent(getEcalDQMSetupObjects(), id));
      tItr->setBinContent(towerid / processedEvents);
    }
    // Fill Channel Status Map MEs
    // Record is checked for updates at every endLumi and filled here
    MESet::iterator chSEnd(meChStatus.end(GetElectronicsMap()));
    for (MESet::iterator chSItr(meChStatus.beginChannel(GetElectronicsMap())); chSItr != chSEnd;
         chSItr.toNextChannel(GetElectronicsMap())) {
      DetId id(chSItr->getId());

      EcalChannelStatusMap::const_iterator chIt(nullptr);

      // Set appropriate channel map (EB or EE)
      if (id.subdetId() == EcalBarrel) {
        EBDetId ebid(id);
        chIt = chStatus->find(ebid);
      } else {
        EEDetId eeid(id);
        chIt = chStatus->find(eeid);
      }

      // Get status code and fill ME
      if (chIt != chStatus->end()) {
        uint16_t code(chIt->getEncodedStatusCode());
        chSItr->setBinContent(code);
      }

    }  // Channel Status Map

    MESet::iterator qEnd(meQuality.end(GetElectronicsMap()));
    MESet::const_iterator occItr(GetElectronicsMap(), sOccupancy);
    for (MESet::iterator qItr(meQuality.beginChannel(GetElectronicsMap())); qItr != qEnd;
         qItr.toNextChannel(GetElectronicsMap())) {
      occItr = qItr;

      DetId id(qItr->getId());

      bool doMask(meQuality.maskMatches(id, mask, statusManager_, GetTrigTowerMap()));

      float entries(occItr->getBinContent());

      float gain(sGain.getBinContent(getEcalDQMSetupObjects(), id));
      float chid(sChId.getBinContent(getEcalDQMSetupObjects(), id));
      float gainswitch(sGainSwitch.getBinContent(getEcalDQMSetupObjects(), id));

      float towerid(sTowerId.getBinContent(getEcalDQMSetupObjects(), id));
      float blocksize(sBlockSize.getBinContent(getEcalDQMSetupObjects(), id));

      if (entries + gain + chid + gainswitch + towerid + blocksize < 1.) {
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMUnknown : kUnknown);
        continue;
      }

      float chErr((gain + chid + gainswitch + towerid + blocksize) /
                  (entries + gain + chid + gainswitch + towerid + blocksize));

      if (chErr > errFractionThreshold_) {
        qItr->setBinContent(doMask ? kMBad : kBad);
        meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMBad : kBad);
      } else {
        qItr->setBinContent(doMask ? kMGood : kGood);
        meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMGood : kGood);
      }
    }

    // Quality check: set an entire FED to BAD if "any" DCC-SRP or DCC-TCC mismatch errors are detected
    // Fill mismatch statistics
    MESet const& sBXSRP(sources_.at("BXSRP"));
    MESet const& sBXTCC(sources_.at("BXTCC"));
    std::vector<bool> hasMismatchDCC(nDCC, false);
    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
      if (sBXSRP.getBinContent(getEcalDQMSetupObjects(), iDCC + 1) > 50. ||
          sBXTCC.getBinContent(getEcalDQMSetupObjects(), iDCC + 1) > 50.)  // "any" => 50
        hasMismatchDCC[iDCC] = true;
    }
    // Analyze mismatch statistics
    for (MESet::iterator qsItr(meQualitySummary.beginChannel(GetElectronicsMap()));
         qsItr != meQualitySummary.end(GetElectronicsMap());
         qsItr.toNextChannel(GetElectronicsMap())) {
      DetId id(qsItr->getId());
      unsigned iDCC(dccId(id, GetElectronicsMap()) - 1);
      if (hasMismatchDCC[iDCC])
        meQualitySummary.setBinContent(
            getEcalDQMSetupObjects(),
            id,
            meQualitySummary.maskMatches(id, mask, statusManager_, GetTrigTowerMap()) ? kMBad : kBad);
    }

  }  // producePlots()

  DEFINE_ECALDQM_WORKER(IntegrityClient);
}  // namespace ecaldqm
