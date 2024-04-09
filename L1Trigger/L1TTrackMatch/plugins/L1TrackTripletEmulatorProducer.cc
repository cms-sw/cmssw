// Original Author:   George Karathanasis,
//                    georgios.karathanasis@cern.ch, CU Boulder
//
//         Created:  Tue, 05 Dec 2023 14:01:41 GMT
//
// Three track candidates (triplets) producer with arbitary track mass. Aimed to
// run in the GTT stage. Triplets are created using the 3 most energetic tracks.
// Selection cuts are applied both on each track individually and in the final
// triplet. Initially created for W->3pi search.
// Link https://indico.cern.ch/event/1356822/contributions/5712593/attachments/2773493/4833005/L1T_w3pi_emulator.pdf

// L1T include files
#include "DataFormats/L1TCorrelator/interface/TkTriplet.h"
#include "DataFormats/L1TCorrelator/interface/TkTripletFwd.h"
#include "DataFormats/L1Trigger/interface/TkTripletWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "DataFormats/Math/interface/LorentzVector.h"

// system include files
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

//own headers
#include "L1TrackUnpacker.h"

//general
#include <ap_int.h>

using namespace std;
using namespace edm;
using namespace l1t;
using namespace l1trackunpacker;

class L1TrackTripletEmulatorProducer : public stream::EDProducer<> {
public:
  explicit L1TrackTripletEmulatorProducer(const ParameterSet &);
  ~L1TrackTripletEmulatorProducer() override = default;
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef vector<L1TTTrackType> L1TTTrackCollectionType;
  typedef edm::RefVector<L1TTTrackCollectionType> L1TTTrackRefCollectionType;
  static void fillDescriptions(ConfigurationDescriptions &descriptions);

private:
  void produce(Event &, const EventSetup &) override;

  // ----------member data ---------------------------

  std::vector<edm::Ptr<L1TTTrackType>> L1TrkPtrs_;
  vector<int> tdtrk_;
  const double trk1_pt_;
  const double trk1_eta_;
  const double trk1_mva_;
  const double trk1_nstub_;
  const double trk1_dz_;
  const double trk1_mass_;
  const double trk2_pt_;
  const double trk2_eta_;
  const double trk2_mva_;
  const double trk2_nstub_;
  const double trk2_dz_;
  const double trk2_mass_;
  const double trk3_pt_;
  const double trk3_eta_;
  const double trk3_mva_;
  const double trk3_nstub_;
  const double trk3_dz_;
  const double trk3_mass_;
  const bool displaced_;
  const double triplet_massMin_;
  const double triplet_massMax_;
  const double triplet_ptMin_;
  const double triplet_ptMax_;
  const double triplet_etaMin_;
  const double triplet_etaMax_;
  const int triplet_abscharge_;
  const double triplet_massOver_;
  const double pair1_massMin_;
  const double pair1_massMax_;
  const double pair2_massMin_;
  const double pair2_massMax_;
  const double pair1_dzMin_;
  const double pair1_dzMax_;
  const double pair2_dzMin_;
  const double pair2_dzMax_;
  const bool use_float_track_precision_;

  struct L1track {
    double Pt;
    double Eta;
    double Phi;
    int Charge;
    double MVA;
    int Nstubs;
    double Z0;
    unsigned int Index;
  };
  //    L1TTTrackType track;
  bool TrackSelector(L1track &, double, double, double, double, double, int);
  double FloatPtFromBits(const L1TTTrackType &);
  double FloatEtaFromBits(const L1TTTrackType &);
  double FloatPhiFromBits(const L1TTTrackType &);
  double FloatZ0FromBits(const L1TTTrackType &);

  const EDGetTokenT<L1TTTrackRefCollectionType> trackToken_;
  const EDGetTokenT<l1t::VertexWordCollection> PVtxToken_;
};

//constructor
L1TrackTripletEmulatorProducer::L1TrackTripletEmulatorProducer(const ParameterSet &iConfig)
    : trk1_pt_(iConfig.getParameter<double>("trk1_ptMin")),
      trk1_eta_(iConfig.getParameter<double>("trk1_absEtaMax")),
      trk1_mva_(iConfig.getParameter<double>("trk1_mvaMin")),
      trk1_nstub_(iConfig.getParameter<int>("trk1_nstubMin")),
      trk1_dz_(iConfig.getParameter<double>("trk1_dzMax")),
      trk1_mass_(iConfig.getParameter<double>("trk1_mass")),
      trk2_pt_(iConfig.getParameter<double>("trk2_ptMin")),
      trk2_eta_(iConfig.getParameter<double>("trk2_absEtaMax")),
      trk2_mva_(iConfig.getParameter<double>("trk2_mvaMin")),
      trk2_nstub_(iConfig.getParameter<int>("trk2_nstubMin")),
      trk2_dz_(iConfig.getParameter<double>("trk2_dzMax")),
      trk2_mass_(iConfig.getParameter<double>("trk2_mass")),
      trk3_pt_(iConfig.getParameter<double>("trk3_ptMin")),
      trk3_eta_(iConfig.getParameter<double>("trk3_absEtaMax")),
      trk3_mva_(iConfig.getParameter<double>("trk3_mvaMin")),
      trk3_nstub_(iConfig.getParameter<int>("trk3_nstubMin")),
      trk3_dz_(iConfig.getParameter<double>("trk3_dzMax")),
      trk3_mass_(iConfig.getParameter<double>("trk3_mass")),
      displaced_(iConfig.getParameter<bool>("displaced")),
      triplet_massMin_(iConfig.getParameter<double>("triplet_massMin")),
      triplet_massMax_(iConfig.getParameter<double>("triplet_massMax")),
      triplet_ptMin_(iConfig.getParameter<double>("triplet_ptMin")),
      triplet_ptMax_(iConfig.getParameter<double>("triplet_ptMax")),
      triplet_etaMin_(iConfig.getParameter<double>("triplet_absEtaMin")),
      triplet_etaMax_(iConfig.getParameter<double>("triplet_absEtaMax")),
      triplet_abscharge_(iConfig.getParameter<int>("triplet_absCharge")),
      triplet_massOver_(iConfig.getParameter<double>("triplet_massOverflow")),
      pair1_massMin_(iConfig.getParameter<double>("pair1_massMin")),
      pair1_massMax_(iConfig.getParameter<double>("pair1_massMax")),
      pair2_massMin_(iConfig.getParameter<double>("pair2_massMin")),
      pair2_massMax_(iConfig.getParameter<double>("pair2_massMax")),
      pair1_dzMin_(iConfig.getParameter<double>("pair1_dzMin")),
      pair1_dzMax_(iConfig.getParameter<double>("pair1_dzMax")),
      pair2_dzMin_(iConfig.getParameter<double>("pair2_dzMin")),
      pair2_dzMax_(iConfig.getParameter<double>("pair2_dzMax")),
      use_float_track_precision_(iConfig.getParameter<bool>("float_precision")),
      trackToken_(consumes<L1TTTrackRefCollectionType>(iConfig.getParameter<InputTag>("L1TrackInputTag"))),
      PVtxToken_(consumes<l1t::VertexWordCollection>(iConfig.getParameter<InputTag>("L1PVertexInputTag"))) {
  if (displaced_) {
    produces<l1t::TkTripletCollection>("L1TrackTripletExtended");
    produces<l1t::TkTripletWordCollection>("L1TrackTripletWordExtended");
  } else {
    produces<l1t::TkTripletCollection>("L1TrackTriplet");
    produces<l1t::TkTripletWordCollection>("L1TrackTripletWord");
  }
}

void L1TrackTripletEmulatorProducer::produce(Event &iEvent, const EventSetup &iSetup) {
  unique_ptr<l1t::TkTripletCollection> L1TrackTripletContainer(new l1t::TkTripletCollection);
  unique_ptr<l1t::TkTripletWordCollection> L1TrackTripletWordContainer(new l1t::TkTripletWordCollection);

  // Read inputs
  edm::Handle<L1TTTrackRefCollectionType> TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);

  edm::Handle<l1t::VertexWordCollection> PVtx;
  iEvent.getByToken(PVtxToken_, PVtx);
  double PVz = (PVtx->at(0)).z0();

  std::string OutputDigisName = "L1TrackTriplet";
  std::string OutputWordName = "L1TrackTripletWord";

  if (displaced_)
    OutputDigisName += "Extended";

  if (TTTrackHandle->size() < 3) {
    iEvent.put(std::move(L1TrackTripletContainer), OutputDigisName);
    iEvent.put(std::move(L1TrackTripletWordContainer), OutputWordName);
    return;
  }

  // Finding the 3 highest pT tracks
  L1track trk1{-99, -99, -99, -99, -99, -99, -99, 0};
  L1track trk2{-99, -99, -99, -99, -99, -99, -99, 0};
  L1track trk3{-99, -99, -99, -99, -99, -99, -99, 0};

  int current_track_idx = -1;
  for (auto current_track : *TTTrackHandle) {
    current_track_idx += 1;
    double current_track_pt = 0;
    if (use_float_track_precision_)
      current_track_pt = current_track->momentum().perp();
    else
      current_track_pt = FloatPtFromBits(*current_track);
    if (current_track_pt < trk1.Pt)
      continue;
    trk3 = trk2;
    trk2 = trk1;
    if (use_float_track_precision_) {
      trk1.Pt = current_track->momentum().perp();
      trk1.Eta = current_track->eta();
      trk1.Phi = current_track->phi();
      trk1.Charge = (int)(current_track->rInv() / fabs(current_track->rInv()));
      trk1.MVA = current_track->trkMVA1();
      trk1.Nstubs = current_track->getStubRefs().size();
      trk1.Z0 = current_track->z0();
      trk1.Index = current_track_idx;
    } else {
      trk1.Pt = FloatPtFromBits(*current_track);
      trk1.Eta = FloatEtaFromBits(*current_track);
      trk1.Phi = FloatPhiFromBits(*current_track);
      trk1.Charge = (int)(current_track->rInv() / fabs(current_track->rInv()));
      trk1.MVA = current_track->trkMVA1();
      trk1.Nstubs = current_track->getStubRefs().size();
      trk1.Z0 = FloatZ0FromBits(*current_track);
      trk1.Index = current_track_idx;
    }
  }

  // track selection
  bool track1_pass = TrackSelector(trk1, PVz, trk1_pt_, trk1_eta_, trk1_mva_, trk1_dz_, trk1_nstub_);
  bool track2_pass = TrackSelector(trk2, PVz, trk2_pt_, trk2_eta_, trk2_mva_, trk2_dz_, trk2_nstub_);
  bool track3_pass = TrackSelector(trk3, PVz, trk3_pt_, trk3_eta_, trk3_mva_, trk3_dz_, trk3_nstub_);
  if (!track1_pass || !track2_pass || !track3_pass) {
    iEvent.put(std::move(L1TrackTripletContainer), OutputDigisName);
    iEvent.put(std::move(L1TrackTripletWordContainer), OutputWordName);
    return;
  }

  // create Lorentz Vectors
  math::PtEtaPhiMLorentzVectorD pion1(trk1.Pt, trk1.Eta, trk1.Phi, trk1_mass_);
  math::PtEtaPhiMLorentzVectorD pion2(trk2.Pt, trk2.Eta, trk2.Phi, trk2_mass_);
  math::PtEtaPhiMLorentzVectorD pion3(trk3.Pt, trk3.Eta, trk3.Phi, trk3_mass_);

  bool event_pass = true;

  // create triplets
  double triplet_mass = (pion1 + pion2 + pion3).M();
  if (triplet_mass > triplet_massOver_)
    triplet_mass = triplet_massOver_;
  double triplet_eta = (pion1 + pion2 + pion3).Eta();
  double triplet_pt = (pion1 + pion2 + pion3).Pt();
  int triplet_charge = trk1.Charge + trk2.Charge + trk3.Charge;
  std::vector<double> pair_masses{(pion1 + pion2).M(), (pion2 + pion3).M(), (pion1 + pion3).M()};
  std::sort(pair_masses.begin(), pair_masses.end(), [](auto &a, auto &b) { return a > b; });
  std::vector<double> pair_dzs{fabs(trk1.Z0 - trk2.Z0), fabs(trk2.Z0 - trk3.Z0), fabs(trk1.Z0 - trk3.Z0)};
  std::sort(pair_dzs.begin(), pair_dzs.end(), [](auto &a, auto &b) { return a > b; });

  //triplet selection
  if (triplet_mass < triplet_massMin_ || triplet_mass > triplet_massMax_)
    event_pass = false;
  if (fabs(triplet_eta) < triplet_etaMin_ || fabs(triplet_eta) > triplet_etaMax_)
    event_pass = false;
  if (triplet_pt < triplet_ptMin_ || triplet_pt > triplet_ptMax_)
    event_pass = false;
  if (fabs(triplet_charge) != triplet_abscharge_ && triplet_abscharge_ > -1)
    event_pass = false;
  if (pair_masses[2] < pair2_massMin_ || pair_masses[2] > pair2_massMax_)
    event_pass = false;
  if (pair_masses[0] < pair1_massMin_ || pair_masses[0] > pair1_massMax_)
    event_pass = false;
  if (pair_dzs[2] < pair2_dzMin_ || pair_dzs[2] > pair2_dzMax_)
    event_pass = false;
  if (pair_dzs[0] < pair1_dzMin_ || pair_dzs[0] > pair1_dzMax_)
    event_pass = false;

  if (!event_pass) {
    iEvent.put(std::move(L1TrackTripletContainer), OutputDigisName);
    iEvent.put(std::move(L1TrackTripletWordContainer), OutputWordName);
    return;
  }
  float tripletPx = (pion1 + pion2 + pion3).Pt() * cos((pion1 + pion2 + pion3).Phi());
  float tripletPy = (pion1 + pion2 + pion3).Pt() * sin((pion1 + pion2 + pion3).Phi());
  float tripletPz = (pion1 + pion2 + pion3).Pt() * sinh((pion1 + pion2 + pion3).Eta());
  float tripletE =
      sqrt(tripletPx * tripletPx + tripletPy * tripletPy + tripletPz * tripletPz + triplet_mass * triplet_mass);
  TkTriplet trkTriplet(math::XYZTLorentzVector(tripletPx, tripletPy, tripletPz, tripletE),
                       triplet_charge,
                       pair_masses[0],
                       pair_masses[2],
                       pair_dzs[0],
                       pair_dzs[2],
                       {edm::Ptr<L1TTTrackType>(TTTrackHandle, trk1.Index),
                        edm::Ptr<L1TTTrackType>(TTTrackHandle, trk2.Index),
                        edm::Ptr<L1TTTrackType>(TTTrackHandle, trk3.Index)});
  L1TrackTripletContainer->push_back(trkTriplet);

  iEvent.put(std::move(L1TrackTripletContainer), OutputDigisName);

  // bit assignment
  l1t::TkTripletWord::valid_t val = 1;
  l1t::TkTripletWord::pt_t bitPt = pt_intern((pion1 + pion2 + pion3).Pt());
  l1t::TkTripletWord::glbeta_t bitEta =
      DoubleToBit((pion1 + pion2 + pion3).Eta(),
                  TkTripletWord::TkTripletBitWidths::kGlbEtaSize,
                  TkTripletWord::MAX_ETA / (1 << TkTripletWord::TkTripletBitWidths::kGlbEtaSize));
  l1t::TkTripletWord::glbphi_t bitPhi =
      DoubleToBit((pion1 + pion2 + pion3).Phi(),
                  TkTripletWord::TkTripletBitWidths::kGlbPhiSize,
                  (2. * std::abs(M_PI)) / (1 << TkTripletWord::TkTripletBitWidths::kGlbPhiSize));
  l1t::TkTripletWord::charge_t bitCharge =
      DoubleToBit(double(triplet_charge),
                  TkTripletWord::TkTripletBitWidths::kChargeSize,
                  TkTripletWord::MAX_CHARGE / (1 << TkTripletWord::TkTripletBitWidths::kChargeSize));
  l1t::TkTripletWord::mass_t bitMass =
      DoubleToBit((pion1 + pion2 + pion3).M(),
                  TkTripletWord::TkTripletBitWidths::kMassSize,
                  TkTripletWord::MAX_MASS / (1 << TkTripletWord::TkTripletBitWidths::kMassSize));
  l1t::TkTripletWord::ditrack_minmass_t bitDiTrackMinMass =
      DoubleToBit(pair_masses[2],
                  TkTripletWord::TkTripletBitWidths::kDiTrackMinMassSize,
                  TkTripletWord::MAX_MASS / (1 << TkTripletWord::TkTripletBitWidths::kDiTrackMinMassSize));
  l1t::TkTripletWord::ditrack_maxmass_t bitDiTrackMaxMass =
      DoubleToBit(pair_masses[0],
                  TkTripletWord::TkTripletBitWidths::kDiTrackMaxMassSize,
                  TkTripletWord::MAX_MASS / (1 << TkTripletWord::TkTripletBitWidths::kDiTrackMaxMassSize));

  l1t::TkTripletWord::ditrack_minz0_t bitDiTrackMinZ0 =
      DoubleToBit(pair_dzs[2],
                  TkTripletWord::TkTripletBitWidths::kDiTrackMinZ0Size,
                  TkTripletWord::MAX_Z0 / (1 << TkTripletWord::TkTripletBitWidths::kDiTrackMinZ0Size));
  l1t::TkTripletWord::ditrack_maxz0_t bitDiTrackMaxZ0 =
      DoubleToBit(pair_dzs[0],
                  TkTripletWord::TkTripletBitWidths::kDiTrackMaxZ0Size,
                  TkTripletWord::MAX_Z0 / (1 << TkTripletWord::TkTripletBitWidths::kDiTrackMaxZ0Size));

  l1t::TkTripletWord::unassigned_t unassigned = 0;
  l1t::TkTripletWord bitTriplet(val,
                                bitPt,
                                bitEta,
                                bitPhi,
                                bitMass,
                                bitCharge,
                                bitDiTrackMinZ0,
                                bitDiTrackMaxMass,
                                bitDiTrackMinZ0,
                                bitDiTrackMaxZ0,
                                unassigned);

  L1TrackTripletWordContainer->push_back(bitTriplet);

  iEvent.put(std::move(L1TrackTripletWordContainer), OutputWordName);
}

bool L1TrackTripletEmulatorProducer::TrackSelector(L1TrackTripletEmulatorProducer::L1track &track,
                                                   double PVz,
                                                   double pt_min_cut,
                                                   double eta_max_cut,
                                                   double mva_min_cut,
                                                   double dz_max_cut,
                                                   int nstub_min_cut) {
  return (track.Pt >= pt_min_cut) && (fabs(track.Eta) <= eta_max_cut) && (track.MVA >= mva_min_cut) &&
         (fabs(track.Z0 - PVz) <= dz_max_cut) && (track.Nstubs >= nstub_min_cut);
}

double L1TrackTripletEmulatorProducer::FloatPtFromBits(const L1TTTrackType &track) {
  ap_uint<TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1> ptBits = track.getRinvWord();
  pt_intern digipt;
  digipt.V = ptBits.range();
  return (double)digipt;
}

double L1TrackTripletEmulatorProducer::FloatEtaFromBits(const L1TTTrackType &track) {
  TTTrack_TrackWord::tanl_t etaBits = track.getTanlWord();
  glbeta_intern digieta;
  digieta.V = etaBits.range();
  return (double)digieta;
}

double L1TrackTripletEmulatorProducer::FloatPhiFromBits(const L1TTTrackType &track) {
  int Sector = track.phiSector();
  double sector_phi_value = 0;
  if (Sector < 5) {
    sector_phi_value = 2.0 * M_PI * Sector / 9.0;
  } else {
    sector_phi_value = (-1.0 * M_PI + M_PI / 9.0 + (Sector - 5) * 2.0 * M_PI / 9.0);
  }
  glbphi_intern trkphiSector = DoubleToBit(
      sector_phi_value, TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit, TTTrack_TrackWord::stepPhi0);
  glbphi_intern local_phiBits = 0;
  local_phiBits.V = track.getPhiWord();

  glbphi_intern local_phi =
      DoubleToBit(BitToDouble(local_phiBits, TTTrack_TrackWord::TrackBitWidths::kPhiSize, TTTrack_TrackWord::stepPhi0),
                  TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit,
                  TTTrack_TrackWord::stepPhi0);
  glbphi_intern digiphi = local_phi + trkphiSector;
  return BitToDouble(
      digiphi, TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit, TTTrack_TrackWord::stepPhi0);
}

double L1TrackTripletEmulatorProducer::FloatZ0FromBits(const L1TTTrackType &track) {
  z0_intern trkZ = track.getZ0Word();
  return BitToDouble(trkZ, TTTrack_TrackWord::TrackBitWidths::kZ0Size, TTTrack_TrackWord::stepZ0);
}

void L1TrackTripletEmulatorProducer::fillDescriptions(ConfigurationDescriptions &descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.add<edm::InputTag>("L1PVertexInputTag", edm::InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"));
  desc.add<double>("trk1_ptMin", -1.0);
  desc.add<double>("trk1_absEtaMax", 10e7);
  desc.add<double>("trk1_mvaMin", -1.0);
  desc.add<int>("trk1_nstubMin", -1);
  desc.add<double>("trk1_dzMax", 10e7);
  desc.add<double>("trk1_mass", 0.139);
  desc.add<double>("trk2_ptMin", -1.0);
  desc.add<double>("trk2_absEtaMax", 10e7);
  desc.add<double>("trk2_mvaMin", -1.0);
  desc.add<int>("trk2_nstubMin", -1);
  desc.add<double>("trk2_dzMax", 10e7);
  desc.add<double>("trk2_mass", 0.139);
  desc.add<double>("trk3_ptMin", -1.0);
  desc.add<double>("trk3_absEtaMax", 10e7);
  desc.add<double>("trk3_mvaMin", -1.0);
  desc.add<int>("trk3_nstubMin", 0);
  desc.add<double>("trk3_dzMax", 10e7);
  desc.add<double>("trk3_mass", 0.139);
  desc.add<bool>("displaced", false);
  desc.add<double>("triplet_massMin", -1.0);
  desc.add<double>("triplet_massMax", 10e7);
  desc.add<double>("triplet_absEtaMin", -1.0);
  desc.add<double>("triplet_absEtaMax", 10e7);
  desc.add<double>("triplet_ptMin", -1.0);
  desc.add<double>("triplet_ptMax", 10e7);
  desc.add<int>("triplet_absCharge", -1);
  desc.add<double>("triplet_massOverflow", 1000);
  desc.add<double>("pair1_massMin", -1);
  desc.add<double>("pair1_massMax", 10e7);
  desc.add<double>("pair2_massMin", -1);
  desc.add<double>("pair2_massMax", 10e7);
  desc.add<double>("pair1_dzMin", -1);
  desc.add<double>("pair1_dzMax", 10e7);
  desc.add<double>("pair2_dzMin", -1);
  desc.add<double>("pair2_dzMax", 10e7);
  desc.add<bool>("float_precision", false);
  descriptions.add("l1tTrackTripletEmulator", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackTripletEmulatorProducer);
