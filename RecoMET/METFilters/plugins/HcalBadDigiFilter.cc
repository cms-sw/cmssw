// -*- C++ -*-
//
// Package:    RecoMET/METFilters
// Class:      HcalBadDigiFilter
//
/**\class HcalBadDigiFilter HcalBadDigiFilter.cc RecoMET/METFilters/plugins/HcalBadDigiFilter.cc

 Description: Set/unset event flags depending on quality of HBHE digis and recHits.
 Mainly needed for >=Run 3.
 More info in https://gitlab.cern.ch/cmshcal/docs/-/work_items/315#note_11812610

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Vinay Hegde
//         Created:  Thu, 02 Jul 2026 14:23:11 GMT
//
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

//
// class declaration
//

using namespace edm;
using namespace std;

class HcalBadDigiFilter : public edm::stream::EDFilter<> {
public:
  explicit HcalBadDigiFilter(const edm::ParameterSet&);
  bool clusteredChannels(const std::vector<DetId> detIds);
  bool isBadRecHit(const uint32_t r_flag, const double r_energy);
  vector<uint32_t> getFlagBits(vector<string>);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  EDGetTokenT<HcalUnpackerReport> unpackerReportLabel_;
  EDGetTokenT<HBHERecHitCollection> hbheRecHitsLabel_;
  ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> chQualityToken_;

  const uint32_t maxBadChannels_;
  const vector<string> listOfFlags_;
  const vector<double> minRecHitEnergies_;
  const bool useBadChannelsTopology_;
  const bool debug_;
  const vector<uint32_t> listOfFlagBits_;

  Handle<HcalUnpackerReport> unpackerReportHandle_;
  Handle<HBHERecHitCollection> hbheRecHitsHandle_;
};

HcalBadDigiFilter::HcalBadDigiFilter(const edm::ParameterSet& iConfig)
    : unpackerReportLabel_(consumes<HcalUnpackerReport>(iConfig.getParameter<edm::InputTag>("unpackerReportLabel"))),
      hbheRecHitsLabel_(consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheRecHitsLabel"))),
      chQualityToken_(esConsumes<HcalChannelQuality, HcalChannelQualityRcd>(edm::ESInputTag("", "withTopo"))),
      maxBadChannels_(iConfig.getParameter<uint32_t>("maxBadChannels")),
      listOfFlags_(iConfig.getParameter<std::vector<std::string>>("listOfFlags")),
      minRecHitEnergies_(iConfig.getParameter<std::vector<double>>("minRecHitEnergies")),
      useBadChannelsTopology_(iConfig.getParameter<bool>("useBadChannelsTopology")),
      debug_(iConfig.getParameter<bool>("debug")),
      listOfFlagBits_(getFlagBits(listOfFlags_)) {
  if (listOfFlagBits_.size() != minRecHitEnergies_.size()) {
    throw cms::Exception("Error") << "Length of minRecHitEnergies = " << minRecHitEnergies_.size()
                                  << ", length of flags = " << listOfFlagBits_.size()
                                  << ". These two should be the same.";
  }
}

// ------------ method called on each new Event  ------------
bool HcalBadDigiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iEvent.getByToken(unpackerReportLabel_, unpackerReportHandle_);
  iEvent.getByToken(hbheRecHitsLabel_, hbheRecHitsHandle_);

  set<DetId> detId_badChn;
  int nBadChn_recHit = 0, nBadChn_unpacker = 0;
  //first check recHit flags and collected bad channels list
  const HBHERecHitCollection* HBHERecHits;
  if (hbheRecHitsHandle_.isValid()) {
    HBHERecHits = hbheRecHitsHandle_.product();
    for (HBHERecHitCollection::const_iterator hbherechit = HBHERecHits->begin(); hbherechit != HBHERecHits->end();
         hbherechit++) {
      uint32_t r_flag = hbherechit->flags();
      double r_energy = hbherechit->energy();
      if (isBadRecHit(r_flag, r_energy)) {
        const DetId id = hbherechit->detid();
        detId_badChn.insert(id);
        nBadChn_recHit++;
      }
    }
  }
  //now check unpacker report
  if (unpackerReportHandle_.isValid()) {
    for (vector<DetId>::const_iterator it = unpackerReportHandle_->bad_quality_begin();
         it != unpackerReportHandle_->bad_quality_end();
         ++it) {
      detId_badChn.insert(it->rawId());
      nBadChn_unpacker++;
    }
  }

  if (detId_badChn.empty())
    return true;  // for most events, this is the case.

  //ignore channels that are already known as bad in DB, if they exist in detId_badChn list
  const HcalChannelQuality& chQuality = iSetup.getData(chQualityToken_);
  vector<DetId> detId_badChnupdated;
  for (const auto& it : detId_badChn) {
    const HcalDetId cell = HcalDetId(it);
    const HcalSubdetector subdet = cell.subdet();
    if (!(subdet == HcalSubdetector::HcalBarrel || subdet == HcalSubdetector::HcalEndcap))
      continue;
    if (chQuality.exists(cell)) {
      const HcalChannelStatus cs = *chQuality.getValues(cell);
      if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
        continue;
    }
    detId_badChnupdated.push_back(it);
  }

  if (!detId_badChnupdated.empty() && debug_)
    edm::LogInfo("HcalBadDigiFilter") << " nBad from unpacker:" << nBadChn_unpacker
                                      << " nBad from recHits:" << nBadChn_recHit
                                      << " nBad after DB check:" << detId_badChnupdated.size() << endl;

  //decide whether to keep the event or not
  if (detId_badChnupdated.size() > maxBadChannels_)
    return false;
  else if (useBadChannelsTopology_ && clusteredChannels(detId_badChnupdated))
    return false;

  return true;
}

// check if required flags are set
bool HcalBadDigiFilter::isBadRecHit(const uint32_t r_flag, const double r_energy) {
  for (size_t i = 0; i < listOfFlagBits_.size(); i++) {
    if (((r_flag >> listOfFlagBits_[i]) & 1) && (r_energy > minRecHitEnergies_[i]))
      return true;
  }
  return false;
}

vector<uint32_t> HcalBadDigiFilter::getFlagBits(vector<string> flagNames) {
  vector<uint32_t> flagBits;
  for (const auto& name : flagNames) {
    if (name == "HBHERun3BadCapId")
      flagBits.push_back(HcalPhase1FlagLabels::HBHERun3BadCapId);
    else if (name == "HBHERun3NonrotatingCapId")
      flagBits.push_back(HcalPhase1FlagLabels::HBHERun3NonrotatingCapId);
    else if (name == "HBHERun3StuckADC")
      flagBits.push_back(HcalPhase1FlagLabels::HBHERun3StuckADC);
    else if (name == "HBHERun3repeatedADCblock")
      flagBits.push_back(HcalPhase1FlagLabels::HBHERun3repeatedADCblock);
    else
      throw cms::Exception("Error") << "Couldn't find the bit index associated to this string: " << name;
  }
  return flagBits;
}
//Find how many channel are neighbors. If nNeighbors is > half of bad channels, they are considered as clustered
bool HcalBadDigiFilter::clusteredChannels(const std::vector<DetId> detIds) {
  uint32_t nNeighbors = 0;
  const std::size_t nCells = detIds.size();
  for (size_t i = 0; i < nCells; i++) {
    HcalDetId hid = HcalDetId(detIds[i]);
    const int ieta = hid.ieta();
    const int iphi = hid.iphi();
    const int idepth = hid.depth();

    for (size_t j = i + 1; j < nCells; j++) {
      HcalDetId hid2 = HcalDetId(detIds[j]);
      if (hid == hid2)
        continue;

      const int jeta = hid2.ieta();
      const int jphi = hid2.iphi();
      const int jdepth = hid2.depth();
      //usually bad channels come in alternate depths. So, use up to next-to-next neighbors in depth
      if ((abs(ieta - jeta) <= 1) && (abs(iphi - jphi) <= 1) && (abs(idepth - jdepth) <= 2)) {
        nNeighbors++;
      }
      if (nNeighbors > nCells / 2)
        return true;
    }
  }
  return false;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalBadDigiFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  desc.add<InputTag>("hbheRecHitsLabel", edm::InputTag("hbhereco"));
  desc.add<InputTag>("unpackerReportLabel", edm::InputTag("hcalDigis"));
  desc.add<bool>("debug", false);
  desc.add<vector<string>>(
      "listOfFlags", {"HBHERun3BadCapId", "HBHERun3NonrotatingCapId", "HBHERun3StuckADC", "HBHERun3repeatedADCblock"});
  desc.add<vector<double>>("minRecHitEnergies", {-100., -100., 10., 10.});
  desc.add<uint32_t>("maxBadChannels", 5);
  desc.add<bool>("useBadChannelsTopology", false);
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HcalBadDigiFilter);
