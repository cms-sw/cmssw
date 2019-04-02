/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "RecoCTPPS/ProtonReconstruction/interface/ProtonReconstructionAlgorithm.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonProducer : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSProtonProducer(const edm::ParameterSet&);
    ~CTPPSProtonProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tracksToken_;

    std::string lhcInfoLabel_;

    unsigned int verbosity_;

    bool doSingleRPReconstruction_;
    bool doMultiRPReconstruction_;

    std::string singleRPReconstructionLabel_;
    std::string multiRPReconstructionLabel_;

    ProtonReconstructionAlgorithm algorithm_;

    bool opticsValid_;
    float currentCrossingAngle_;
};

//----------------------------------------------------------------------------------------------------

CTPPSProtonProducer::CTPPSProtonProducer(const edm::ParameterSet& iConfig) :
  tracksToken_                (consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("tagLocalTrackLite"))),
  lhcInfoLabel_               (iConfig.getParameter<std::string>("lhcInfoLabel")),
  verbosity_                  (iConfig.getUntrackedParameter<unsigned int>("verbosity", 0)),
  doSingleRPReconstruction_   (iConfig.getParameter<bool>("doSingleRPReconstruction")),
  doMultiRPReconstruction_    (iConfig.getParameter<bool>("doMultiRPReconstruction")),
  singleRPReconstructionLabel_(iConfig.getParameter<std::string>("singleRPReconstructionLabel")),
  multiRPReconstructionLabel_ (iConfig.getParameter<std::string>("multiRPReconstructionLabel")),
  algorithm_                  (iConfig.getParameter<bool>("fitVtxY"), iConfig.getParameter<bool>("useImprovedInitialEstimate"), verbosity_),
  opticsValid_(false),
  currentCrossingAngle_(-1.)
{
  if (doSingleRPReconstruction_)
    produces<reco::ForwardProtonCollection>(singleRPReconstructionLabel_);

  if (doMultiRPReconstruction_)
    produces<reco::ForwardProtonCollection>(multiRPReconstructionLabel_);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("tagLocalTrackLite", edm::InputTag("ctppsLocalTrackLiteProducer"))
    ->setComment("specification of the input lite-track collection");

  desc.add<std::string>("lhcInfoLabel", "")
    ->setComment("label of the LHCInfo record");

  desc.addUntracked<unsigned int>("verbosity", 0)->setComment("verbosity level");

  desc.add<bool>("doSingleRPReconstruction", true)
    ->setComment("flag whether to apply single-RP reconstruction strategy");

  desc.add<bool>("doMultiRPReconstruction", true)
    ->setComment("flag whether to apply multi-RP reconstruction strategy");

  desc.add<std::string>("singleRPReconstructionLabel", "singleRP")
    ->setComment("output label for single-RP reconstruction products");

  desc.add<std::string>("multiRPReconstructionLabel", "multiRP")
    ->setComment("output label for multi-RP reconstruction products");

  desc.add<bool>("fitVtxY", true)
    ->setComment("for multi-RP reconstruction, flag whether y* should be free fit parameter");

  desc.add<bool>("useImprovedInitialEstimate", true)
    ->setComment("for multi-RP reconstruction, flag whether a quadratic estimate of the initial point should be used");

  descriptions.add("ctppsProtons", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get conditions
  edm::ESHandle<LHCInfo> hLHCInfo;
  iSetup.get<LHCInfoRcd>().get(lhcInfoLabel_, hLHCInfo);

  edm::ESHandle<LHCInterpolatedOpticalFunctionsSetCollection> hOpticalFunctions;
  iSetup.get<CTPPSInterpolatedOpticsRcd>().get(hOpticalFunctions);

  // re-initialise algorithm upon crossing-angle change
  if (hLHCInfo->crossingAngle() != currentCrossingAngle_) {
    currentCrossingAngle_ = hLHCInfo->crossingAngle();

    if (hOpticalFunctions->empty()) {
      edm::LogInfo("CTPPSProtonProducer") << "No optical functions available, reconstruction disabled.";
      algorithm_.release();
      opticsValid_ = false;
    }
    else {
      algorithm_.init(*hOpticalFunctions);
      opticsValid_ = true;
    }
  }

  // book output
  std::unique_ptr<reco::ForwardProtonCollection> pOutSingleRP(new reco::ForwardProtonCollection);
  std::unique_ptr<reco::ForwardProtonCollection> pOutMultiRP(new reco::ForwardProtonCollection);

  // do reconstruction only if optics is valid
  if (opticsValid_)
  {
    // prepare log
    std::ostringstream ssLog;
    if (verbosity_)
      ssLog << "input tracks:";

    // get input
    edm::Handle<CTPPSLocalTrackLiteCollection> hTracks;
    iEvent.getByToken(tracksToken_, hTracks);

    // keep only tracks from tracker RPs, split them by LHC sector
    CTPPSLocalTrackLiteRefVector tracks_45, tracks_56;
    std::map<CTPPSDetId, unsigned int> nTracksPerRP;
    for (unsigned int idx = 0; idx < hTracks->size(); ++idx) {
      const auto& tr = hTracks->at(idx);
      const CTPPSDetId rpId(tr.getRPId());
      if (rpId.subdetId() != CTPPSDetId::sdTrackingStrip && rpId.subdetId() != CTPPSDetId::sdTrackingPixel)
        continue;

      if (verbosity_)
        ssLog << "\n\t"
          << tr.getRPId() << " (" << (rpId.arm()*100 + rpId.station()*10 + rpId.rp()) << "): "
          << "x=" << tr.getX() << " +- " << tr.getXUnc() << " mm, "
          << "y=" << tr.getY() << " +- " << tr.getYUnc() << " mm";

      CTPPSLocalTrackLiteRef r_track(hTracks, idx);
      if (rpId.arm() == 0)
        tracks_45.push_back(r_track);
      if (rpId.arm() == 1)
        tracks_56.push_back(r_track);

      nTracksPerRP[rpId]++;
    }

    // for the moment: check whether there is no more than 1 track in each arm
    bool singleTrack_45 = true, singleTrack_56 = true;
    for (const auto& detid_num : nTracksPerRP) {
      if (detid_num.second > 1) {
        const CTPPSDetId& rpId = detid_num.first;
        if (rpId.arm() == 0)
          singleTrack_45 = false;
        if (rpId.arm() == 1)
          singleTrack_56 = false;
      }
    }

    // single-RP reconstruction
    if (doSingleRPReconstruction_) {
      algorithm_.reconstructFromSingleRP(tracks_45, *pOutSingleRP, *hLHCInfo, ssLog);
      algorithm_.reconstructFromSingleRP(tracks_56, *pOutSingleRP, *hLHCInfo, ssLog);
    }

    // multi-RP reconstruction
    if (doMultiRPReconstruction_) {
      if (singleTrack_45)
        algorithm_.reconstructFromMultiRP(tracks_45, *pOutMultiRP, *hLHCInfo, ssLog);
      if (singleTrack_56)
        algorithm_.reconstructFromMultiRP(tracks_56, *pOutMultiRP, *hLHCInfo, ssLog);
    }

    // dump log
    if (verbosity_)
      edm::LogInfo("CTPPSProtonProducer") << ssLog.str();
  }

  // save output
  if (doSingleRPReconstruction_)
    iEvent.put(std::move(pOutSingleRP), singleRPReconstructionLabel_);

  if (doMultiRPReconstruction_)
    iEvent.put(std::move(pOutMultiRP), multiRPReconstructionLabel_);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonProducer);

