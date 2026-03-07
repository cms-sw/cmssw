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
#include "L1Trigger/L1TTrackMatch/interface/TkTripletEmuAlgo.h"

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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
    l1ttripletemu::pt_t f_Pt;
    l1ttripletemu::eta_t f_Eta;
    l1ttripletemu::global_phi_t globalPhi;
    TTTrack_TrackWord::phi_t localPhi;
    int phiSector;
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

  // Firmware-relevant members
  std::vector<l1ttripletemu::cos_lut_fixed_t> cosLUT_;    // Cos LUT array
  std::vector<l1ttripletemu::cosh_lut_fixed_t> coshLUT_;  // Cosh LUT array
  std::vector<l1ttripletemu::sinh_lut_fixed_t> sinhLUT_;  // Sinh LUT array
  std::vector<l1ttripletemu::global_phi_t> phiQuadrants_;
  std::vector<l1ttripletemu::global_phi_t> phiShifts_;
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
    phiQuadrants_ = l1ttripletemu::generatePhiSliceLUT(l1ttripletemu::kNQuadrants);
    phiShifts_ = l1ttripletemu::generatePhiSliceLUT(l1ttripletemu::kNSector);

    // Compute LUTs
    cosLUT_ = l1ttripletemu::generateCosLUT();
    coshLUT_ = l1ttripletemu::generateCoshLUT();
    sinhLUT_ = l1ttripletemu::generateSinhLUT();

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

  L1track trk1{0, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 0};
  L1track trk2{0, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 0};
  L1track trk3{0, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 0};

  l1ttripletemu::tktriplet_mass_sq_t f_tktriplet_mass_sq = 0;
  int current_track_idx = -1;

  // Loop over L1 tracks in event
  for (auto current_track : *TTTrackHandle) {
    double current_track_pt = FloatPtFromBits(*current_track);
    current_track_idx += 1;

    // Select three highest-pT tracks and store relevant quantities
    if (current_track_pt > (double)trk1.Pt) {
      trk3 = trk2;
      trk2 = trk1;
      trk1.f_Pt = (l1ttripletemu::pt_t)FloatPtFromBits(*current_track);
      trk1.f_Eta = (l1ttripletemu::eta_t)FloatEtaFromBits(*current_track);
      trk1.globalPhi =
          l1ttripletemu::localToGlobalPhi(current_track->getPhiWord(), phiShifts_[current_track->phiSector()]);
      trk1.phiSector = current_track->phiSector();
      trk1.localPhi = (TTTrack_TrackWord::phi_t)current_track->getPhiWord();
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
    } else if (current_track_pt > (double)trk2.Pt) {
      trk3 = trk2;
      trk2.f_Pt = (l1ttripletemu::pt_t)FloatPtFromBits(*current_track);
      trk2.f_Eta = (l1ttripletemu::eta_t)FloatEtaFromBits(*current_track);
      trk2.globalPhi =
          l1ttripletemu::localToGlobalPhi(current_track->getPhiWord(), phiShifts_[current_track->phiSector()]);
      trk2.localPhi = (TTTrack_TrackWord::phi_t)current_track->getPhiWord();
      trk2.phiSector = current_track->phiSector();
      if (use_float_track_precision_) {
        trk2.Pt = current_track->momentum().perp();
        trk2.Eta = current_track->eta();
        trk2.Phi = current_track->phi();
        trk2.Charge = (int)(current_track->rInv() / fabs(current_track->rInv()));
        trk2.MVA = current_track->trkMVA1();
        trk2.Nstubs = current_track->getStubRefs().size();
        trk2.Z0 = current_track->z0();
        trk2.Index = current_track_idx;
      } else {
        trk2.Pt = FloatPtFromBits(*current_track);
        trk2.Eta = FloatEtaFromBits(*current_track);
        trk2.Phi = FloatPhiFromBits(*current_track);
        trk2.Charge = (int)(current_track->rInv() / fabs(current_track->rInv()));
        trk2.MVA = current_track->trkMVA1();
        trk2.Nstubs = current_track->getStubRefs().size();
        trk2.Z0 = FloatZ0FromBits(*current_track);
        trk2.Index = current_track_idx;
      }
    } else if (current_track_pt > (double)trk3.Pt) {
      trk3.f_Pt = (l1ttripletemu::pt_t)FloatPtFromBits(*current_track);
      trk3.f_Eta = (l1ttripletemu::eta_t)FloatEtaFromBits(*current_track);
      trk3.globalPhi =
          l1ttripletemu::localToGlobalPhi(current_track->getPhiWord(), phiShifts_[current_track->phiSector()]);
      trk3.localPhi = (TTTrack_TrackWord::phi_t)current_track->getPhiWord();
      trk3.phiSector = current_track->phiSector();
      if (use_float_track_precision_) {
        trk3.Pt = current_track->momentum().perp();
        trk3.Eta = current_track->eta();
        trk3.Phi = current_track->phi();
        trk3.Charge = (int)(current_track->rInv() / fabs(current_track->rInv()));
        trk3.MVA = current_track->trkMVA1();
        trk3.Nstubs = current_track->getStubRefs().size();
        trk3.Z0 = current_track->z0();
        trk3.Index = current_track_idx;
      } else {
        trk3.Pt = FloatPtFromBits(*current_track);
        trk3.Eta = FloatEtaFromBits(*current_track);
        trk3.Phi = FloatPhiFromBits(*current_track);
        trk3.Charge = (int)(current_track->rInv() / fabs(current_track->rInv()));
        trk3.MVA = current_track->trkMVA1();
        trk3.Nstubs = current_track->getStubRefs().size();
        trk3.Z0 = FloatZ0FromBits(*current_track);
        trk3.Index = current_track_idx;
      }
    }
  }

  // Triplet invariant mass calculation (moved OUTSIDE the track loop)
  // Check that all triplet tracks are valid
  if (trk1.f_Pt == 0 || trk2.f_Pt == 0 || trk3.f_Pt == 0) {
    iEvent.put(std::move(L1TrackTripletContainer), OutputDigisName);
    iEvent.put(std::move(L1TrackTripletWordContainer), OutputWordName);
    return;
  }

  // Define sinh LUT indices
  const l1ttripletemu::sinh_lut_index_t sinhIndex1 =
      (l1ttripletemu::sinh_lut_index_t)((std::abs((float)trk1.f_Eta)) /
                                            (2.5 / (1 << l1ttripletemu::kSinhLUTTableSize)) +
                                        1);
  const l1ttripletemu::sinh_lut_index_t sinhIndex2 =
      (l1ttripletemu::sinh_lut_index_t)((std::abs((float)trk2.f_Eta)) /
                                            (2.5 / (1 << l1ttripletemu::kSinhLUTTableSize)) +
                                        1);
  const l1ttripletemu::sinh_lut_index_t sinhIndex3 =
      (l1ttripletemu::sinh_lut_index_t)((std::abs((float)trk3.f_Eta)) /
                                            (2.5 / (1 << l1ttripletemu::kSinhLUTTableSize)) +
                                        1);

  // Define cosh LUT indices
  const l1ttripletemu::cosh_lut_index_t coshIndex1 =
      (l1ttripletemu::cosh_lut_index_t)((std::abs((float)trk1.f_Eta)) /
                                            (2.5 / (1 << l1ttripletemu::kCoshLUTTableSize)) +
                                        1);
  const l1ttripletemu::cosh_lut_index_t coshIndex2 =
      (l1ttripletemu::cosh_lut_index_t)((std::abs((float)trk2.f_Eta)) /
                                            (2.5 / (1 << l1ttripletemu::kCoshLUTTableSize)) +
                                        1);
  const l1ttripletemu::cosh_lut_index_t coshIndex3 =
      (l1ttripletemu::cosh_lut_index_t)((std::abs((float)trk3.f_Eta)) /
                                            (2.5 / (1 << l1ttripletemu::kCoshLUTTableSize)) +
                                        1);

  // Total track momenta
  l1ttripletemu::pxyz_t p1 = (l1ttripletemu::pxyz_t)trk1.f_Pt * coshLUT_[coshIndex1];
  l1ttripletemu::pxyz_t p2 = (l1ttripletemu::pxyz_t)trk2.f_Pt * coshLUT_[coshIndex2];
  l1ttripletemu::pxyz_t p3 = (l1ttripletemu::pxyz_t)trk3.f_Pt * coshLUT_[coshIndex3];

  // Z-component track momenta
  l1ttripletemu::pxyz_t pz1 = (l1ttripletemu::pxyz_t)trk1.f_Pt * sinhLUT_[sinhIndex1];
  l1ttripletemu::pxyz_t pz2 = (l1ttripletemu::pxyz_t)trk2.f_Pt * sinhLUT_[sinhIndex2];
  l1ttripletemu::pxyz_t pz3 = (l1ttripletemu::pxyz_t)trk3.f_Pt * sinhLUT_[sinhIndex3];

  // Correct pz sign if eta is negative
  if (trk1.f_Eta < 0) {
    pz1 = -1 * pz1;
  }
  if (trk2.f_Eta < 0) {
    pz2 = -1 * pz2;
  }
  if (trk3.f_Eta < 0) {
    pz3 = -1 * pz3;
  }

  // Momentum x,y component definitions
  l1ttripletemu::pxyz_t px1 = 0, py1 = 0;
  l1ttripletemu::pxyz_t px2 = 0, py2 = 0;
  l1ttripletemu::pxyz_t px3 = 0, py3 = 0;
  L1track tmp_trk{0, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 0};

  // W mass calculation using LUTs (as in firmware)
  for (int tk = 0; tk < 3; tk++) {
    if (tk == 0) {
      tmp_trk = trk1;
    } else if (tk == 1) {
      tmp_trk = trk2;
    } else if (tk == 2) {
      tmp_trk = trk3;
    }

    // Compute px, py for triplet tracks
    if (tmp_trk.globalPhi >= phiQuadrants_[0] && tmp_trk.globalPhi < phiQuadrants_[1]) {
      const l1ttripletemu::cos_lut_index_t cosIndex = (tmp_trk.globalPhi) >> l1ttripletemu::kCosLUTShift;
      const l1ttripletemu::cos_lut_index_t sinIndex =
          (phiQuadrants_[1] - 1 - tmp_trk.globalPhi) >> l1ttripletemu::kCosLUTShift;
      if (tk == 0) {
        px1 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex];
        py1 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex];
      } else if (tk == 1) {
        px2 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex];
        py2 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex];
      } else if (tk == 2) {
        px3 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex];
        py3 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex];
      }
    } else if (tmp_trk.globalPhi >= phiQuadrants_[1] && tmp_trk.globalPhi < phiQuadrants_[2]) {
      const l1ttripletemu::cos_lut_index_t cosIndex =
          (phiQuadrants_[2] - 1 - tmp_trk.globalPhi) >> l1ttripletemu::kCosLUTShift;
      const l1ttripletemu::cos_lut_index_t sinIndex =
          (tmp_trk.globalPhi - phiQuadrants_[1]) >> l1ttripletemu::kCosLUTShift;
      if (tk == 0) {
        px1 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex]);
        py1 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex];
      } else if (tk == 1) {
        px2 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex]);
        py2 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex];
      } else if (tk == 2) {
        px3 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex]);
        py3 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex];
      }
    } else if (tmp_trk.globalPhi >= phiQuadrants_[2] && tmp_trk.globalPhi < phiQuadrants_[3]) {
      const l1ttripletemu::cos_lut_index_t cosIndex =
          (tmp_trk.globalPhi - phiQuadrants_[2]) >> l1ttripletemu::kCosLUTShift;
      const l1ttripletemu::cos_lut_index_t sinIndex =
          (phiQuadrants_[3] - 1 - tmp_trk.globalPhi) >> l1ttripletemu::kCosLUTShift;
      if (tk == 0) {
        px1 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex]);
        py1 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex]);
      } else if (tk == 1) {
        px2 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex]);
        py2 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex]);
      } else if (tk == 2) {
        px3 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex]);
        py3 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex]);
      }
    } else if (tmp_trk.globalPhi >= phiQuadrants_[3] && tmp_trk.globalPhi < phiQuadrants_[4]) {
      const l1ttripletemu::cos_lut_index_t cosIndex =
          (phiQuadrants_[4] - 1 - tmp_trk.globalPhi) >> l1ttripletemu::kCosLUTShift;
      const l1ttripletemu::cos_lut_index_t sinIndex =
          (tmp_trk.globalPhi - phiQuadrants_[3]) >> l1ttripletemu::kCosLUTShift;
      if (tk == 0) {
        px1 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex];
        py1 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex]);
      } else if (tk == 1) {
        px2 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex];
        py2 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex]);
      } else if (tk == 2) {
        px3 = (l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[cosIndex];
        py3 = -((l1ttripletemu::pxyz_t)tmp_trk.f_Pt * cosLUT_[sinIndex]);
      }
    }
  }

  // W mass calculation
  f_tktriplet_mass_sq =
      (l1ttripletemu::tktriplet_mass_sq_t)((p1 + p2 + p3) * (p1 + p2 + p3) - (px1 + px2 + px3) * (px1 + px2 + px3) -
                                           (py1 + py2 + py3) * (py1 + py2 + py3) -
                                           (pz1 + pz2 + pz3) * (pz1 + pz2 + pz3));

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

  double triplet_eta = (pion1 + pion2 + pion3).Eta();
  bool event_pass = true;

  // create triplets
  double triplet_mass = (pion1 + pion2 + pion3).M();
  if (triplet_mass > triplet_massOver_)
    triplet_mass = triplet_massOver_;

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

  // Test vector outputs
  l1t::TkTripletWord::tktriplet_valid_t val = 1;
  l1ttripletemu::TkTriplet tkTriplet;
  tkTriplet.pt = 0;
  tkTriplet.phi = 0;
  tkTriplet.eta = 0;
  tkTriplet.mass = (l1ttripletemu::tktriplet_mass_t)std::sqrt((float)f_tktriplet_mass_sq);
  tkTriplet.trk1Pt = (l1ttripletemu::tktriplet_trk_pt_t)trk1.f_Pt;
  tkTriplet.trk2Pt = (l1ttripletemu::tktriplet_trk_pt_t)trk2.f_Pt;
  tkTriplet.trk3Pt = (l1ttripletemu::tktriplet_trk_pt_t)trk3.f_Pt;
  tkTriplet.charge = 0;
  l1t::TkTripletWord::tktriplet_unassigned_t unassigned = 0;

  l1t::TkTripletWord L1Triplet(val,
                               tkTriplet.pt,
                               tkTriplet.phi,
                               tkTriplet.eta,
                               tkTriplet.mass,
                               tkTriplet.trk1Pt,
                               tkTriplet.trk2Pt,
                               tkTriplet.trk3Pt,
                               tkTriplet.charge,
                               unassigned);

  L1TrackTripletWordContainer->push_back(L1Triplet);

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
  //The following says we do not know what parameters are allowed
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
