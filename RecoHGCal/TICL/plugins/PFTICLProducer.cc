// This producer converts a list of TICLCandidates to a list of PFCandidates.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

class PFTICLProducer : public edm::stream::EDProducer<> {
public:
  PFTICLProducer(const edm::ParameterSet&);
  ~PFTICLProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // parameters
  const bool useMTDTiming_;
  const bool useTimingAverage_;
  const float timingQualityThreshold_;
  const bool energy_from_regression_;
  // inputs
  const edm::EDGetTokenT<edm::View<TICLCandidate>> ticl_candidates_;
  const edm::EDGetTokenT<edm::ValueMap<float>> srcTrackTime_, srcTrackTimeError_, srcTrackTimeQuality_;
  const edm::EDGetTokenT<reco::MuonCollection> muons_;
  // For PFMuonAlgo
  std::unique_ptr<PFMuonAlgo> pfmu_;
};

DEFINE_FWK_MODULE(PFTICLProducer);

PFTICLProducer::PFTICLProducer(const edm::ParameterSet& conf)
    : useMTDTiming_(conf.getParameter<bool>("useMTDTiming")),
      useTimingAverage_(conf.getParameter<bool>("useTimingAverage")),
      timingQualityThreshold_(conf.getParameter<double>("timingQualityThreshold")),
      energy_from_regression_(conf.getParameter<bool>("energyFromRegression")),
      ticl_candidates_(consumes<edm::View<TICLCandidate>>(conf.getParameter<edm::InputTag>("ticlCandidateSrc"))),
      srcTrackTime_(consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("trackTimeValueMap"))),
      srcTrackTimeError_(consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("trackTimeErrorMap"))),
      srcTrackTimeQuality_(consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("trackTimeQualityMap"))),
      muons_(consumes<reco::MuonCollection>(conf.getParameter<edm::InputTag>("muonSrc"))),
      pfmu_(std::make_unique<PFMuonAlgo>(conf.getParameterSet("pfMuonAlgoParameters"),
                                         false)) {  // postMuonCleaning = false
  produces<reco::PFCandidateCollection>();
}

void PFTICLProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ticlCandidateSrc", edm::InputTag("ticlTrackstersMerge"));
  desc.add<edm::InputTag>("trackTimeValueMap", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("trackTimeErrorMap", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("trackTimeQualityMap", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<bool>("energyFromRegression", true);
  desc.add<double>("timingQualityThreshold", 0.5);
  desc.add<bool>("useMTDTiming", true);
  desc.add<bool>("useTimingAverage", false);
  // For PFMuonAlgo
  desc.add<edm::InputTag>("muonSrc", edm::InputTag("muons1stStep"));
  edm::ParameterSetDescription psd_PFMuonAlgo;
  PFMuonAlgo::fillPSetDescription(psd_PFMuonAlgo);
  desc.add<edm::ParameterSetDescription>("pfMuonAlgoParameters", psd_PFMuonAlgo);
  //
  descriptions.add("pfTICLProducer", desc);
}

void PFTICLProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  //get TICLCandidates
  edm::Handle<edm::View<TICLCandidate>> ticl_cand_h;
  evt.getByToken(ticl_candidates_, ticl_cand_h);
  const auto ticl_candidates = *ticl_cand_h;
  edm::Handle<edm::ValueMap<float>> trackTimeH, trackTimeErrH, trackTimeQualH;
  evt.getByToken(srcTrackTime_, trackTimeH);
  evt.getByToken(srcTrackTimeError_, trackTimeErrH);
  evt.getByToken(srcTrackTimeQuality_, trackTimeQualH);
  const auto muonH = evt.getHandle(muons_);
  const auto& muons = *muonH;

  auto candidates = std::make_unique<reco::PFCandidateCollection>();

  for (const auto& ticl_cand : ticl_candidates) {
    const auto abs_pdg_id = std::abs(ticl_cand.pdgId());
    const auto charge = ticl_cand.charge();
    const auto& four_mom = ticl_cand.p4();
    float total_raw_energy = 0.f;
    float total_em_raw_energy = 0.f;
    for (const auto& t : ticl_cand.tracksters()) {
      total_raw_energy += t->raw_energy();
      total_em_raw_energy += t->raw_em_energy();
    }
    float ecal_energy_fraction = total_em_raw_energy / total_raw_energy;
    float ecal_energy = energy_from_regression_ ? ticl_cand.p4().energy() * ecal_energy_fraction
                                                : ticl_cand.rawEnergy() * ecal_energy_fraction;
    float hcal_energy =
        energy_from_regression_ ? ticl_cand.p4().energy() - ecal_energy : ticl_cand.rawEnergy() - ecal_energy;
    // fix for floating point rounding could go slightly below 0
    hcal_energy = std::max(0.f, hcal_energy);
    reco::PFCandidate::ParticleType part_type;
    switch (abs_pdg_id) {
      case 11:
        part_type = reco::PFCandidate::e;
        break;
      case 13:
        part_type = reco::PFCandidate::mu;
        break;
      case 22:
        part_type = reco::PFCandidate::gamma;
        break;
      case 130:
        part_type = reco::PFCandidate::h0;
        break;
      case 211:
        part_type = reco::PFCandidate::h;
        break;
      // default also handles neutral pions (111) for the time being (not yet foreseen in PFCandidate)
      default:
        part_type = reco::PFCandidate::X;
    }

    candidates->emplace_back(charge, four_mom, part_type);

    auto& candidate = candidates->back();
    candidate.setEcalEnergy(ecal_energy, ecal_energy);
    candidate.setHcalEnergy(hcal_energy, hcal_energy);
    if (candidate.charge()) {  // otherwise PFCandidate throws
      // Construct edm::Ref from edm::Ptr. As of now, assumes type to be reco::Track. To be extended (either via
      // dynamic type checking or configuration) if additional track types are needed.
      reco::TrackRef trackref(ticl_cand.trackPtr().id(), int(ticl_cand.trackPtr().key()), &evt.productGetter());
      candidate.setTrackRef(trackref);
      // Utilize PFMuonAlgo
      const int muId = PFMuonAlgo::muAssocToTrack(trackref, muons);
      if (muId != -1) {
        const reco::MuonRef muonref = reco::MuonRef(muonH, muId);
        const bool allowLoose = (part_type == reco::PFCandidate::mu);
        // Redefine pfmuon candidate kinematics and add muonref
        pfmu_->reconstructMuon(candidate, muonref, allowLoose);
      }
    }

    // HGCAL timing as default values
    auto time = ticl_cand.time();
    auto timeE = ticl_cand.timeError();

    if (useMTDTiming_ and candidate.charge()) {
      // Ignore HGCAL timing until it will be TOF corrected
      time = -99.;
      timeE = -1.;
      // Check MTD timing availability
      const bool assocQuality = (*trackTimeQualH)[candidate.trackRef()] > timingQualityThreshold_;
      if (assocQuality) {
        const auto timeHGC = time;
        const auto timeEHGC = timeE;
        const auto timeMTD = (*trackTimeH)[candidate.trackRef()];
        const auto timeEMTD = (*trackTimeErrH)[candidate.trackRef()];

        if (useTimingAverage_ && (timeEMTD > 0 && timeEHGC > 0)) {
          // Compute weighted average between HGCAL and MTD timing
          const auto invTimeESqHGC = pow(timeEHGC, -2);
          const auto invTimeESqMTD = pow(timeEMTD, -2);
          timeE = (invTimeESqHGC * invTimeESqMTD) / (invTimeESqHGC + invTimeESqMTD);
          time = (timeHGC * invTimeESqHGC + timeMTD * invTimeESqMTD) * timeE;
          timeE = sqrt(timeE);
        } else if (timeEMTD > 0) {  // Ignore HGCal timing until it will be TOF corrected
          time = timeMTD;
          timeE = timeEMTD;
        }
      }
    }
    candidate.setTime(time, timeE);
  }

  evt.put(std::move(candidates));
}
