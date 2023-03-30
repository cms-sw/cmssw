/****************************************************************************
 *
 * This is a part of PPS offline software.
 * Authors:
 *   Edoardo Bossini
 *   Piotr Maciej Cwiklicki
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "CalibPPS/TimingCalibration/interface/TimingCalibrationStruct.h"

//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLWorker : public DQMGlobalEDAnalyzer<TimingCalibrationHistograms> {
public:
  explicit PPSTimingCalibrationPCLWorker(const edm::ParameterSet&);

  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const TimingCalibrationHistograms&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&,
                      const edm::Run&,
                      const edm::EventSetup&,
                      TimingCalibrationHistograms&) const override;

  template <typename T>
  bool searchForProduct(edm::Event const& iEvent,
                        const std::vector<edm::EDGetTokenT<T>>& tokens,
                        const std::vector<edm::InputTag>& tags,
                        edm::Handle<T>& handle) const;

  const std::vector<edm::InputTag> RecHitTags_;
  std::vector<edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondRecHit>>> diamondRecHitTokens_;
  const edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geomEsToken_;

  const std::string dqmDir_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLWorker::PPSTimingCalibrationPCLWorker(const edm::ParameterSet& iConfig)
    : RecHitTags_(iConfig.getParameter<std::vector<edm::InputTag>>("diamondRecHitTags")),
      geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      dqmDir_(iConfig.getParameter<std::string>("dqmDir")) {
  for (auto& tag : RecHitTags_)
    diamondRecHitTokens_.push_back(consumes<edm::DetSetVector<CTPPSDiamondRecHit>>(tag));
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::bookHistograms(DQMStore::IBooker& iBooker,
                                                   const edm::Run& iRun,
                                                   const edm::EventSetup& iSetup,
                                                   TimingCalibrationHistograms& iHists) const {
  iBooker.cd();
  iBooker.setCurrentFolder(dqmDir_);
  std::string ch_name;

  const auto& geom = iSetup.getData(geomEsToken_);
  for (auto it = geom.beginSensor(); it != geom.endSensor(); ++it) {
    if (!CTPPSDiamondDetId::check(it->first))
      continue;
    const CTPPSDiamondDetId detid(it->first);

    detid.channelName(ch_name);
    iHists.leadingTime[detid.rawId()] = iBooker.book1D("t_" + ch_name, ch_name + ";t (ns);Entries", 1200, -60., 60.);
    iHists.toT[detid.rawId()] = iBooker.book1D("tot_" + ch_name, ch_name + ";ToT (ns);Entries", 100, -20., 20.);
    iHists.leadingTimeVsToT[detid.rawId()] =
        iBooker.book2D("tvstot_" + ch_name, ch_name + ";ToT (ns);t (ns)", 240, 0., 60., 450, -20., 25.);
  }
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::dqmAnalyze(const edm::Event& iEvent,
                                               const edm::EventSetup& iSetup,
                                               const TimingCalibrationHistograms& iHists) const {
  edm::Handle<edm::DetSetVector<CTPPSDiamondRecHit>> dsv_rechits;
  // then extract the rechits information for later processing
  searchForProduct(iEvent, diamondRecHitTokens_, RecHitTags_, dsv_rechits);

  // ensure timing detectors rechits are found in the event content
  if (dsv_rechits->empty()) {
    edm::LogWarning("PPSTimingCalibrationPCLWorker:dqmAnalyze") << "No rechits retrieved from the event content.";
    return;
  }
  for (const auto& ds_rechits : *dsv_rechits) {
    const CTPPSDiamondDetId detid(ds_rechits.detId());
    if (iHists.leadingTimeVsToT.count(detid.rawId()) == 0) {
      edm::LogWarning("PPSTimingCalibrationPCLWorker:dqmAnalyze")
          << "Pad with detId=" << detid << " is not set to be monitored.";
      continue;
    }
    for (const auto& rechit : ds_rechits) {
      // skip invalid rechits
      if (rechit.time() == 0. || rechit.toT() < 0.)
        continue;
      iHists.leadingTime.at(detid.rawId())->Fill(rechit.time());
      iHists.toT.at(detid.rawId())->Fill(rechit.toT());
      iHists.leadingTimeVsToT.at(detid.rawId())->Fill(rechit.toT(), rechit.time());
    }
  }
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("diamondRecHitTags", {edm::InputTag("ctppsDiamondRecHits")})
      ->setComment("input tag for the PPS diamond detectors rechits");
  desc.add<std::string>("dqmDir", "AlCaReco/PPSTimingCalibrationPCL")
      ->setComment("output path for the various DQM plots");

  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
bool PPSTimingCalibrationPCLWorker::searchForProduct(edm::Event const& iEvent,
                                                     const std::vector<edm::EDGetTokenT<T>>& tokens,
                                                     const std::vector<edm::InputTag>& tags,
                                                     edm::Handle<T>& handle) const {
  bool foundProduct = false;
  for (unsigned int i = 0; i < tokens.size(); i++)
    if (auto h = iEvent.getHandle(tokens[i])) {
      handle = h;
      foundProduct = true;
      edm::LogInfo("searchForProduct") << "Found a product with " << tags[i];
      break;
    }

  if (!foundProduct)
    throw edm::Exception(edm::errors::ProductNotFound) << "Could not find a product with any of the selected labels.";

  return foundProduct;
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLWorker);
