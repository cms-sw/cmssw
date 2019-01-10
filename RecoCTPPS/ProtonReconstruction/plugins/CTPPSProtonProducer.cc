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

#include "DataFormats/ProtonReco/interface/ProtonTrackFwd.h"

#include "RecoCTPPS/ProtonReconstruction/interface/ProtonReconstructionAlgorithm.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsCollection.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonProducer : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSProtonProducer(const edm::ParameterSet&);
    ~CTPPSProtonProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    edm::EDGetTokenT<std::vector<CTPPSLocalTrackLite> > tracksToken_;

    unsigned int verbosity_;

    bool doSingleRPReconstruction_;
    bool doMultiRPReconstruction_;

    std::string singleRPReconstructionLabel_;
    std::string multiRPReconstructionLabel_;

    ProtonReconstructionAlgorithm algorithm_;

    edm::ESWatcher<LHCInfoRcd> lhcInfoWatcher_;
    float currentCrossingAngle_;

    std::unordered_map<unsigned int, LHCOpticalFunctionsSet> opticalFunctions_;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSProtonProducer::CTPPSProtonProducer(const edm::ParameterSet& iConfig) :
  tracksToken_(consumes<std::vector<CTPPSLocalTrackLite> >(iConfig.getParameter<edm::InputTag>("tagLocalTrackLite"))),
  verbosity_               (iConfig.getUntrackedParameter<unsigned int>("verbosity", 0)),
  doSingleRPReconstruction_(iConfig.getParameter<bool>("doSingleRPReconstruction")),
  doMultiRPReconstruction_ (iConfig.getParameter<bool>("doMultiRPReconstruction")),
  singleRPReconstructionLabel_(iConfig.getParameter<std::string>("singleRPReconstructionLabel")),
  multiRPReconstructionLabel_(iConfig.getParameter<std::string>("multiRPReconstructionLabel")),
  algorithm_               (iConfig.getParameter<bool>("fitVtxY"), iConfig.getParameter<bool>("useImprovedInitialEstimate"), verbosity_),
  currentCrossingAngle_(-1.)
{
  if (doSingleRPReconstruction_) {
    produces<reco::ProtonTrackCollection>(singleRPReconstructionLabel_);
    produces<reco::ProtonTrackExtraCollection>(singleRPReconstructionLabel_);
  }

  if (doMultiRPReconstruction_) {
    produces<reco::ProtonTrackCollection>(multiRPReconstructionLabel_);
    produces<reco::ProtonTrackExtraCollection>(multiRPReconstructionLabel_);
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonProducer::fillDescriptions(ConfigurationDescriptions& descriptions)
{
  ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 0)->setComment("verbosity level");
  desc.add<edm::InputTag>("tagLocalTrackLite", edm::InputTag("ctppsLocalTrackLiteProducer"))
    ->setComment("specification of the input lite-track collection");

  desc.add<bool>("doSingleRPReconstruction", true)
    ->setComment("flag whether to apply single-RP reconstruction strategy");
  desc.add<std::string>("singleRPReconstructionLabel", "singleRP")
    ->setComment("output label for single-RP reconstruction products");

  desc.add<bool>("doMultiRPReconstruction", true)
    ->setComment("flag whether to apply multi-RP reconstruction strategy");
  desc.add<std::string>("multiRPReconstructionLabel", "multiRP")
    ->setComment("output label for multi-RP reconstruction products");

  desc.add<bool>("fitVtxY", true)
    ->setComment("for multi-RP reconstruction, flag whether y* should be free fit parameter");
  desc.add<bool>("useImprovedInitialEstimate", true)
    ->setComment("for multi-RP reconstruction, flag whether a quadratic estimate of the initial point should be used");

  descriptions.add("ctppsProtons", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonProducer::produce(Event& event, const EventSetup &eventSetup)
{
  // get conditions
  edm::ESHandle<LHCInfo> hLHCInfo;
  eventSetup.get<LHCInfoRcd>().get(hLHCInfo);

  edm::ESHandle<LHCOpticalFunctionsCollection> hOpticalFunctionCollection;
  eventSetup.get<CTPPSOpticsRcd>().get(hOpticalFunctionCollection);

  // re-initialise algorithm upon crossing-angle change
  if (lhcInfoWatcher_.check(eventSetup))
  {
    const LHCInfo* pLHCInfo = hLHCInfo.product();
    if (pLHCInfo->crossingAngle() != currentCrossingAngle_)
    {
      currentCrossingAngle_ = pLHCInfo->crossingAngle();

      if (currentCrossingAngle_ == 0.)
      {
        LogWarning("CTPPSProtonProducer") << "Invalid crossing angle, reconstruction disabled.";
        algorithm_.release();
      } else {
        if (verbosity_)
          edm::LogInfo("CTPPSProtonProducer") << "Setting crossing angle " << currentCrossingAngle_;

        // interpolate optical functions
        opticalFunctions_.clear();
        hOpticalFunctionCollection->interpolateFunctions(currentCrossingAngle_, opticalFunctions_);
        for (auto &p : opticalFunctions_)
          p.second.initializeSplines();

        // reinitialise algorithm
        algorithm_.init(opticalFunctions_);
      }
    }
  }

  // prepare log
  std::ostringstream ssLog;
  if (verbosity_)
    ssLog << "input tracks:";

  // get input
  Handle<std::vector<CTPPSLocalTrackLite> > hTracks;
  event.getByToken(tracksToken_, hTracks);

  // keep only tracks from tracker RPs, split them by LHC sector
  reco::ProtonTrackExtra::CTPPSLocalTrackLiteRefVector tracks_45, tracks_56;
  map<CTPPSDetId, unsigned int> nTracksPerRP;
  for (unsigned int idx = 0; idx < hTracks->size(); ++idx) {
    const CTPPSLocalTrackLite &tr = (*hTracks)[idx];
    CTPPSDetId rpId(tr.getRPId());
    if (rpId.subdetId() != CTPPSDetId::sdTrackingStrip && rpId.subdetId() != CTPPSDetId::sdTrackingPixel)
      continue;

    if (verbosity_)
      ssLog << "\n"
        << "    " << tr.getRPId() << " (" << (rpId.arm()*100 + rpId.station()*10 + rpId.rp()) << "): "
        << "x = " << tr.getX() << " +- " << tr.getXUnc() << " mm"
        << ", y=" << tr.getY() << " +- " << tr.getYUnc() << " mm";

    if (rpId.arm() == 0)
      tracks_45.emplace_back(hTracks, idx);
    if (rpId.arm() == 1)
      tracks_56.emplace_back(hTracks, idx);

    nTracksPerRP[rpId]++;
  }

  // for the moment: check whether there is no more than 1 track in each arm
  bool singleTrack_45 = true, singleTrack_56 = true;
  for (const auto& detid_num : nTracksPerRP) {
    if (detid_num.second > 1) {
      const CTPPSDetId &rpId = detid_num.first;
      if (rpId.arm() == 0)
        singleTrack_45 = false;
      if (rpId.arm() == 1)
        singleTrack_56 = false;
    }
  }

  // single-RP reconstruction
  if (doSingleRPReconstruction_) {
    unique_ptr<reco::ProtonTrackCollection> output(new reco::ProtonTrackCollection);

    algorithm_.reconstructFromSingleRP(tracks_45, *output, *outputExtra, *hLHCInfo, ssLog);
    algorithm_.reconstructFromSingleRP(tracks_56, *output, *outputExtra, *hLHCInfo, ssLog);

    auto ohExtra = event.put(move(outputExtra), singleRPReconstructionLabel_);

    for (unsigned int i = 0; i < output->size(); ++i)
      (*output)[i].setProtonTrackExtra(reco::ProtonTrackExtraRef(ohExtra, i));

    event.put(move(output), singleRPReconstructionLabel_);
  }

  // multi-RP reconstruction
  if (doMultiRPReconstruction_) {
    unique_ptr<reco::ProtonTrackCollection> output(new reco::ProtonTrackCollection);

    if (singleTrack_45)
      algorithm_.reconstructFromMultiRP(tracks_45, *output, *outputExtra, *hLHCInfo, ssLog);
    if (singleTrack_56)
      algorithm_.reconstructFromMultiRP(tracks_56, *output, *outputExtra, *hLHCInfo, ssLog);

    auto ohExtra = event.put(move(outputExtra), multiRPReconstructionLabel_);

    for (unsigned int i = 0; i < output->size(); ++i)
      (*output)[i].setProtonTrackExtra(reco::ProtonTrackExtraRef(ohExtra, i));

    event.put(move(output), multiRPReconstructionLabel_);
  }

  if (verbosity_)
    edm::LogInfo("CTPPSProtonProducer") << ssLog.str();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonProducer);
