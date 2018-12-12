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
#include "DataFormats/ProtonReco/interface/ProtonTrackExtraFwd.h"

#include "RecoCTPPS/ProtonReconstruction/interface/ProtonReconstructionAlgorithm.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsCollection.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstruction : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSProtonReconstruction(const edm::ParameterSet&);
    ~CTPPSProtonReconstruction() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&) override;

    edm::EDGetTokenT<std::vector<CTPPSLocalTrackLite> > tracksToken_;

    unsigned int verbosity_;

    bool doSingleRPReconstruction_;
    bool doMultiRPReconstruction_;

    ProtonReconstructionAlgorithm algorithm_;

    edm::ESWatcher<LHCInfoRcd> lhcInfoWatcher_;
    float currentCrossingAngle_;

    std::unordered_map<unsigned int, LHCOpticalFunctionsSet> opticalFunctions_;

    const std::string singleRPLabel_ = "singleRP";
    const std::string multiRPLabel_ = "multiRP";
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstruction::CTPPSProtonReconstruction(const edm::ParameterSet& iConfig) :
  tracksToken_(consumes<std::vector<CTPPSLocalTrackLite> >(iConfig.getParameter<edm::InputTag>("tagLocalTrackLite"))),
  verbosity_               (iConfig.getUntrackedParameter<unsigned int>("verbosity", 0)),
  doSingleRPReconstruction_(iConfig.getParameter<bool>("doSingleRPReconstruction")),
  doMultiRPReconstruction_ (iConfig.getParameter<bool>("doMultiRPReconstruction")),
  algorithm_               (iConfig.getParameter<bool>("fitVtxY"), verbosity_),
  currentCrossingAngle_(-1.)
{
  if (doSingleRPReconstruction_) {
    produces<reco::ProtonTrackCollection>(singleRPLabel_);
    produces<reco::ProtonTrackExtraCollection>(singleRPLabel_);
  }

  if (doMultiRPReconstruction_) {
    produces<reco::ProtonTrackCollection>(multiRPLabel_);
    produces<reco::ProtonTrackExtraCollection>(multiRPLabel_);
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstruction::fillDescriptions(ConfigurationDescriptions& descriptions)
{
  ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 0)->setComment("verbosity level");
  desc.add<edm::InputTag>("tagLocalTrackLite", edm::InputTag("ctppsLocalTrackLiteProducer"))
    ->setComment("specification of the input lite-track collection");
  desc.add<bool>("doSingleRPReconstruction", true)
    ->setComment("flag whether to apply single-RP reconstruction strategy");
  desc.add<bool>("doMultiRPReconstruction", true)
    ->setComment("flag whether to apply multi-RP reconstruction strategy");
  desc.add<bool>("fitVtxY", true)
    ->setComment("for multi-RP reconstruction, flag whether y* should be free fit parameter");

  descriptions.add("ctppsProtonReconstruction", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstruction::produce(Event& event, const EventSetup &eventSetup)
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
        LogWarning("CTPPSProtonReconstruction") << "Invalid crossing angle, reconstruction disabled.";
        algorithm_.release();
      }

      if (verbosity_)
        edm::LogInfo("CTPPSProtonReconstruction") << "Setting crossing angle " << currentCrossingAngle_;

      // interpolate optical functions
      opticalFunctions_.clear();
      hOpticalFunctionCollection->interpolateFunctions(currentCrossingAngle_, opticalFunctions_);
      for (auto &p : opticalFunctions_)
        p.second.initializeSplines();

      // reinitialise algorithm
      algorithm_.init(opticalFunctions_);
    }
  }

  // get input
  Handle<std::vector<CTPPSLocalTrackLite> > hTracks;
  event.getByToken(tracksToken_, hTracks);

  // keep only tracks from tracker RPs, split them by LHC sector
  reco::ProtonTrackExtra::CTPPSLocalTrackLiteRefVector tracks_45, tracks_56;
  map<unsigned int, unsigned int> nTracksPerRP;
  for (unsigned int idx = 0; idx < hTracks->size(); ++idx) {
    const CTPPSLocalTrackLite &tr = (*hTracks)[idx];
    CTPPSDetId rpId(tr.getRPId());
    if (rpId.subdetId() != CTPPSDetId::sdTrackingStrip && rpId.subdetId() != CTPPSDetId::sdTrackingPixel)
      continue;

    if (verbosity_)
      LogDebug("CTPPSProtonReconstruction")
        << "    " << tr.getRPId() << " (" << (rpId.arm()*100 + rpId.station()*10 + rpId.rp()) << "): "
        << "x = " << tr.getX() << " +- " << tr.getXUnc() << " mm"
        << ", y=" << tr.getY() << " +- " << tr.getYUnc() << " mm" << std::endl;

    if (rpId.arm() == 0)
      tracks_45.emplace_back(hTracks, idx);
    if (rpId.arm() == 1)
      tracks_56.emplace_back(hTracks, idx);

    nTracksPerRP[tr.getRPId()]++;
  }

  // for the moment: check whether there is no more than 1 track in each arm
  bool singleTrack_45 = true, singleTrack_56 = true;
  for_each(nTracksPerRP.begin(), nTracksPerRP.end(), [&singleTrack_45,&singleTrack_56](const auto& p) {
    if (p.second > 1) {
      CTPPSDetId rpId(p.first);
      if (rpId.arm() == 0)
        singleTrack_45 = false;
      if (rpId.arm() == 1)
        singleTrack_56 = false;
    }
  });

  // single-RP reconstruction
  if (doSingleRPReconstruction_) {
    unique_ptr<reco::ProtonTrackCollection> output(new reco::ProtonTrackCollection);
    unique_ptr<reco::ProtonTrackExtraCollection> outputExtra(new reco::ProtonTrackExtraCollection);

    algorithm_.reconstructFromSingleRP(tracks_45, *output, *outputExtra, *hLHCInfo);
    algorithm_.reconstructFromSingleRP(tracks_56, *output, *outputExtra, *hLHCInfo);

    auto ohExtra = event.put(move(outputExtra), singleRPLabel_);

    for (unsigned int i = 0; i < output->size(); ++i)
      (*output)[i].setProtonTrackExtra(reco::ProtonTrackExtraRef(ohExtra, i));

    event.put(move(output), singleRPLabel_);
  }

  // multi-RP reconstruction
  if (doMultiRPReconstruction_) {
    unique_ptr<reco::ProtonTrackCollection> output(new reco::ProtonTrackCollection);
    unique_ptr<reco::ProtonTrackExtraCollection> outputExtra(new reco::ProtonTrackExtraCollection);

    if (singleTrack_45)
      algorithm_.reconstructFromMultiRP(tracks_45, *output, *outputExtra, *hLHCInfo);
    if (singleTrack_56)
      algorithm_.reconstructFromMultiRP(tracks_56, *output, *outputExtra, *hLHCInfo);

    auto ohExtra = event.put(move(outputExtra), multiRPLabel_);

    for (unsigned int i = 0; i < output->size(); ++i)
      (*output)[i].setProtonTrackExtra(reco::ProtonTrackExtraRef(ohExtra, i));

    event.put(move(output), multiRPLabel_);
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstruction);

