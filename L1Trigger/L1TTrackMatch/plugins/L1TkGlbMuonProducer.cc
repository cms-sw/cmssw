// input: L1TkTracks and  L1Muon
// match the two and produce a collection of L1TkGlbMuonParticle
// eventually, this should be made modular and allow to swap out different algorithms

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1TCorrelator/interface/TkGlbMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkGlbMuonFwd.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"
#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

// system include files
#include <memory>
#include <string>

static constexpr float mu_mass = 0.105658369;
static constexpr float dr2_cutoff = 0.3;
static constexpr float matching_factor_eta = 3.;
static constexpr float matching_factor_phi = 4.;
static constexpr float min_mu_propagator_p = 3.5;
static constexpr float min_mu_propagator_barrel_pT = 3.5;
static constexpr float max_mu_propagator_eta = 2.5;

using namespace l1t;

class L1TkGlbMuonProducer : public edm::global::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  struct PropState {  //something simple, imagine it's hardware emulation
    PropState() : pt(-99), eta(-99), phi(-99), sigmaPt(-99), sigmaEta(-99), sigmaPhi(-99), valid(false) {}
    float pt;
    float eta;
    float phi;
    float sigmaPt;
    float sigmaEta;
    float sigmaPhi;
    bool valid;
  };

  enum AlgoType { kTP = 1, kDynamicWindows = 2, kMantra = 3 };

  explicit L1TkGlbMuonProducer(const edm::ParameterSet&);
  ~L1TkGlbMuonProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  PropState propagateToGMT(const L1TTTrackType& l1tk) const;
  double sigmaEtaTP(const Muon& mu) const;
  double sigmaPhiTP(const Muon& mu) const;

  // the TP algorithm
  void runOnMuonCollection_v1(const edm::Handle<MuonBxCollection>&,
                              const edm::Handle<L1TTTrackCollectionType>&,
                              TkGlbMuonCollection& tkMuons) const;

  float etaMin_;
  float etaMax_;
  float etaBO_;  //eta value for barrel-overlap fontier
  float etaOE_;  //eta value for overlap-endcap fontier
  float zMax_;   // |z_track| < zMax_ in cm
  float chi2Max_;
  float pTMinTra_;
  float dRMax_;
  int nStubsmin_;  // minimum number of stubs
  bool correctGMTPropForTkZ_;
  bool use5ParameterFit_;
  bool useTPMatchWindows_;

  const edm::EDGetTokenT<MuonBxCollection> muToken;
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > trackToken_;
};

L1TkGlbMuonProducer::L1TkGlbMuonProducer(const edm::ParameterSet& iConfig)
    : etaMin_((float)iConfig.getParameter<double>("ETAMIN")),
      etaMax_((float)iConfig.getParameter<double>("ETAMAX")),
      zMax_((float)iConfig.getParameter<double>("ZMAX")),
      chi2Max_((float)iConfig.getParameter<double>("CHI2MAX")),
      pTMinTra_((float)iConfig.getParameter<double>("PTMINTRA")),
      dRMax_((float)iConfig.getParameter<double>("DRmax")),
      nStubsmin_(iConfig.getParameter<int>("nStubsmin")),
      muToken(consumes<MuonBxCollection>(iConfig.getParameter<edm::InputTag>("L1MuonInputTag"))),
      trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(
          iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))) {
  correctGMTPropForTkZ_ = iConfig.getParameter<bool>("correctGMTPropForTkZ");

  use5ParameterFit_ = iConfig.getParameter<bool>("use5ParameterFit");
  useTPMatchWindows_ = iConfig.getParameter<bool>("useTPMatchWindows");
  produces<TkGlbMuonCollection>();
}

L1TkGlbMuonProducer::~L1TkGlbMuonProducer() {}

// ------------ method called to produce the data  ------------
void L1TkGlbMuonProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // the L1Mu objects
  edm::Handle<MuonBxCollection> l1musH;
  iEvent.getByToken(muToken, l1musH);

  // the L1Tracks
  edm::Handle<L1TTTrackCollectionType> l1tksH;
  iEvent.getByToken(trackToken_, l1tksH);

  auto tkMuons = std::make_unique<TkGlbMuonCollection>();

  // Fill the collection
  runOnMuonCollection_v1(l1musH, l1tksH, *tkMuons);

  // put the new track+muon objects in the event!
  iEvent.put(std::move(tkMuons));
};

void L1TkGlbMuonProducer::runOnMuonCollection_v1(const edm::Handle<MuonBxCollection>& muonH,
                                                 const edm::Handle<L1TTTrackCollectionType>& l1tksH,
                                                 TkGlbMuonCollection& tkMuons) const {
  const L1TTTrackCollectionType& l1tks = (*l1tksH.product());
  const MuonBxCollection& l1mus = (*muonH.product());

  int imu = 0;

  for (auto l1mu = l1mus.begin(0); l1mu != l1mus.end(0); ++l1mu) {  // considering BX = only
    edm::Ref<MuonBxCollection> l1muRef(muonH, imu);
    imu++;

    float l1mu_eta = l1mu->eta();
    float l1mu_phi = l1mu->phi();

    float l1mu_feta = std::abs(l1mu_eta);
    if (l1mu_feta < etaMin_)
      continue;
    if (l1mu_feta > etaMax_)
      continue;

    float drmin = 999;

    PropState matchProp;
    int match_idx = -1;
    int il1tk = -1;

    int nTracksMatch = 0;

    for (const auto& l1tk : l1tks) {
      il1tk++;

      float l1tk_pt = l1tk.momentum().perp();
      if (l1tk_pt < pTMinTra_)
        continue;

      float l1tk_z = l1tk.POCA().z();
      if (std::abs(l1tk_z) > zMax_)
        continue;

      float l1tk_chi2 = l1tk.chi2();
      if (l1tk_chi2 > chi2Max_)
        continue;

      int l1tk_nstubs = l1tk.getStubRefs().size();
      if (l1tk_nstubs < nStubsmin_)
        continue;

      float l1tk_eta = l1tk.momentum().eta();
      float l1tk_phi = l1tk.momentum().phi();

      float dr2 = reco::deltaR2(l1mu_eta, l1mu_phi, l1tk_eta, l1tk_phi);
      if (dr2 > dr2_cutoff)
        continue;

      nTracksMatch++;

      const PropState& pstate = propagateToGMT(l1tk);
      if (!pstate.valid)
        continue;

      float dr2prop = reco::deltaR2(l1mu_eta, l1mu_phi, pstate.eta, pstate.phi);
      // FIXME: check if this matching procedure can be improved with
      // a pT dependent dR window
      if (dr2prop < drmin) {
        drmin = dr2prop;
        match_idx = il1tk;
        matchProp = pstate;
      }
    }  // over l1tks

    LogDebug("L1TkGlbMuonProducer") << "matching index is " << match_idx;
    if (match_idx >= 0) {
      const L1TTTrackType& matchTk = l1tks[match_idx];

      float sigmaEta = sigmaEtaTP(*l1mu);
      float sigmaPhi = sigmaPhiTP(*l1mu);

      float etaCut = matching_factor_eta * sqrt(sigmaEta * sigmaEta + matchProp.sigmaEta * matchProp.sigmaEta);
      float phiCut = matching_factor_phi * sqrt(sigmaPhi * sigmaPhi + matchProp.sigmaPhi * matchProp.sigmaPhi);

      float dEta = std::abs(matchProp.eta - l1mu_eta);
      float dPhi = std::abs(deltaPhi(matchProp.phi, l1mu_phi));

      bool matchCondition = useTPMatchWindows_ ? dEta < etaCut && dPhi < phiCut : drmin < dRMax_;

      if (matchCondition) {
        edm::Ptr<L1TTTrackType> l1tkPtr(l1tksH, match_idx);

        const auto& p3 = matchTk.momentum();
        float p4e = sqrt(mu_mass * mu_mass + p3.mag2());

        math::XYZTLorentzVector l1tkp4(p3.x(), p3.y(), p3.z(), p4e);

        const auto& tkv3 = matchTk.POCA();
        math::XYZPoint v3(tkv3.x(), tkv3.y(), tkv3.z());  // why is this defined?

        float trkisol = -999;

        TkGlbMuon l1tkmu(l1tkp4, l1muRef, l1tkPtr, trkisol);

        l1tkmu.setTrkzVtx((float)tkv3.z());
        l1tkmu.setdR(drmin);
        l1tkmu.setNTracksMatched(nTracksMatch);
        tkMuons.push_back(l1tkmu);
      }
    }
  }  //over l1mus
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TkGlbMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

L1TkGlbMuonProducer::PropState L1TkGlbMuonProducer::propagateToGMT(const L1TkGlbMuonProducer::L1TTTrackType& tk) const {
  auto p3 = tk.momentum();
  float tk_pt = p3.perp();
  float tk_p = p3.mag();
  float tk_eta = p3.eta();
  float tk_aeta = std::abs(tk_eta);
  float tk_phi = p3.phi();
  float tk_q = tk.rInv() > 0 ? 1. : -1.;
  float tk_z = tk.POCA().z();
  if (!correctGMTPropForTkZ_)
    tk_z = 0;

  L1TkGlbMuonProducer::PropState dest;
  if (tk_p < min_mu_propagator_p)
    return dest;
  if (tk_aeta < 1.1 && tk_pt < min_mu_propagator_barrel_pT)
    return dest;
  if (tk_aeta > max_mu_propagator_eta)
    return dest;

  //0th order:
  dest.valid = true;

  float dzCorrPhi = 1.;
  float deta = 0;
  float etaProp = tk_aeta;

  if (tk_aeta < 1.1) {
    etaProp = 1.1;
    deta = tk_z / 550. / cosh(tk_aeta);
  } else {
    float delta = tk_z / 850.;  //roughly scales as distance to 2nd station
    if (tk_eta > 0)
      delta *= -1;
    dzCorrPhi = 1. + delta;

    float zOzs = tk_z / 850.;
    if (tk_eta > 0)
      deta = zOzs / (1. - zOzs);
    else
      deta = zOzs / (1. + zOzs);
    deta = deta * tanh(tk_eta);
  }
  float resPhi = tk_phi - 1.464 * tk_q * cosh(1.7) / cosh(etaProp) / tk_pt * dzCorrPhi - M_PI / 144.;
  resPhi = reco::reduceRange(resPhi);

  dest.eta = tk_eta + deta;
  dest.phi = resPhi;
  dest.pt = tk_pt;  //not corrected for eloss

  dest.sigmaEta = 0.100 / tk_pt;  //multiple scattering term
  dest.sigmaPhi = 0.106 / tk_pt;  //need a better estimate for these
  return dest;
}

double L1TkGlbMuonProducer::sigmaEtaTP(const Muon& l1mu) const {
  float l1mu_eta = l1mu.eta();
  if (std::abs(l1mu_eta) <= 1.55)
    return 0.0288;
  else if (std::abs(l1mu_eta) > 1.55 && std::abs(l1mu_eta) <= 1.65)
    return 0.025;
  else if (std::abs(l1mu_eta) > 1.65 && std::abs(l1mu_eta) <= 2.4)
    return 0.0144;
  return 0.0288;
}

double L1TkGlbMuonProducer::sigmaPhiTP(const Muon& mu) const { return 0.0126; }

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkGlbMuonProducer);
