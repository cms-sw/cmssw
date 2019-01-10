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

class CTPPSProtonReconstruction : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSProtonReconstruction(const edm::ParameterSet&);
    ~CTPPSProtonReconstruction() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

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
  algorithm_               (iConfig.getParameter<bool>("fitVtxY"), iConfig.getParameter<bool>("useImprovedInitialEstimate"), verbosity_),
  currentCrossingAngle_(-1.)
{
  if (doSingleRPReconstruction_)
    produces<reco::ProtonTrackCollection>(singleRPLabel_);

  if (doMultiRPReconstruction_)
    produces<reco::ProtonTrackCollection>(multiRPLabel_);
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
  desc.add<bool>("useImprovedInitialEstimate", true)
    ->setComment("for multi-RP reconstruction, flag whether a quadratic estimate of the initial point should be used");

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
      } else {
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
  }

  // prepare log
  std::ostringstream ssLog;
  if (verbosity_)
    ssLog << "input tracks:";

  // get input
  Handle<std::vector<CTPPSLocalTrackLite> > hTracks;
  event.getByToken(tracksToken_, hTracks);

  // keep only tracks from tracker RPs, split them by LHC sector
  reco::ProtonTrack::CTPPSLocalTrackLiteRefVector tracks_45, tracks_56;
  map<unsigned int, unsigned int> nTracksPerRP;
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

    nTracksPerRP[tr.getRPId()]++;
  }

  // for the moment: check whether there is no more than 1 track in each arm
  bool singleTrack_45 = true, singleTrack_56 = true;
  for (const auto& detid_num : nTracksPerRP) {
    if (detid_num.second > 1) {
      CTPPSDetId rpId(detid_num.first);
      if (rpId.arm() == 0)
        singleTrack_45 = false;
      if (rpId.arm() == 1)
        singleTrack_56 = false;
    }
  }

  // single-RP reconstruction
  if (doSingleRPReconstruction_) {
    unique_ptr<reco::ProtonTrackCollection> output(new reco::ProtonTrackCollection);

    algorithm_.reconstructFromSingleRP(tracks_45, *output, *hLHCInfo, ssLog);
    algorithm_.reconstructFromSingleRP(tracks_56, *output, *hLHCInfo, ssLog);

    event.put(move(output), singleRPLabel_);
  }

  // multi-RP reconstruction
  if (doMultiRPReconstruction_) {
    unique_ptr<reco::ProtonTrackCollection> output(new reco::ProtonTrackCollection);

    if (singleTrack_45)
      algorithm_.reconstructFromMultiRP(tracks_45, *output, *hLHCInfo, ssLog);
    if (singleTrack_56)
      algorithm_.reconstructFromMultiRP(tracks_56, *output, *hLHCInfo, ssLog);

    event.put(move(output), multiRPLabel_);
  }

  if (verbosity_)
    edm::LogInfo("CTPPSProtonReconstruction") << ssLog.str();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstruction);

