#ifndef L1Trigger_L1TTrackMatch_L1TkEtMissEmuTrackTransform_HH
#define L1Trigger_L1TTrackMatch_L1TkEtMissEmuTrackTransform_HH

#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"

/*
** class  : L1TkEtMissEmuTrackTransform
** author : Christopher Brown
** date   : 19/02/2021
** modified :16/06/2021
** brief  : Converts TTrack_trackword to internal Et word including vertex

**        : 
*/

// Internal Word used by EtMiss Emulation, producer expects this wordtype
struct InternalEtWord {
  l1tmetemu::z_t pV;
  l1tmetemu::z_t z0;

  l1tmetemu::pt_t pt;
  l1tmetemu::eta_t eta;
  l1tmetemu::global_phi_t globalPhi;
  l1tmetemu::nstub_t nstubs;

  TTTrack_TrackWord::hit_t Hitpattern;
  TTTrack_TrackWord::bendChi2_t bendChi2;
  TTTrack_TrackWord::chi2rphi_t chi2rphidof;
  TTTrack_TrackWord::chi2rz_t chi2rzdof;

  unsigned int Sector;  //Phi sector
  bool EtaSector;       //Positve or negative eta

  float phi;  // Used to debug cos phi LUT
};

class L1TkEtMissEmuTrackTransform {
public:
  L1TkEtMissEmuTrackTransform() = default;
  ~L1TkEtMissEmuTrackTransform() = default;

  void generateLUTs();  // Generate internal LUTs needed for track transfrom

  // Transform track and vertex, allow for vertex word or vertex collections
  template <class track, class vertex>
  InternalEtWord transformTrack(track& track_ref, vertex& PV);

  // Converts local int phi to global int phi
  l1tmetemu::global_phi_t localToGlobalPhi(TTTrack_TrackWord::phi_t local_phi, l1tmetemu::global_phi_t sector_shift);

  // Function to count stubs in hitpattern
  l1tmetemu::nstub_t countNStub(TTTrack_TrackWord::hit_t Hitpattern);

  std::vector<l1tmetemu::global_phi_t> generatePhiSliceLUT(unsigned int N);

  std::vector<l1tmetemu::global_phi_t> getPhiQuad() const { return phiQuadrants; }
  std::vector<l1tmetemu::global_phi_t> getPhiShift() const { return phiShift; }

  void setGTTinput(bool input) { GTTinput_ = input; }

private:
  std::vector<l1tmetemu::global_phi_t> phiQuadrants;
  std::vector<l1tmetemu::global_phi_t> phiShift;

  bool GTTinput_ = false;
};

// Template to allow vertex word or vertex from vertex finder depending on simulation vs emulation
template <class track, class vertex>
InternalEtWord L1TkEtMissEmuTrackTransform::transformTrack(track& track_ref, vertex& PV) {
  InternalEtWord Outword;

  unsigned int temp_pt;
  unsigned int temp_eta;

  if (GTTinput_) {
    if ((track_ref.getRinvWord() & (1 << (TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1))) != 0) {
      // Only Want Magnitude of Pt for sums so perform absolute value
      temp_pt = abs((1 << (TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1)) - track_ref.getRinvWord());
    } else {
      temp_pt = track_ref.getRinvWord();
    }
    Outword.pt = l1tmetemu::transformSignedValue(
        temp_pt * 2, TTTrack_TrackWord::TrackBitWidths::kRinvSize, l1tmetemu::kInternalPtWidth);

    if ((track_ref.getTanlWord() & (1 << (TTTrack_TrackWord::TrackBitWidths::kTanlSize - 1))) != 0) {
      // Only Want Magnitude of Eta for cuts and track to vertex association so
      // perform absolute value
      temp_eta = abs((1 << (TTTrack_TrackWord::TrackBitWidths::kTanlSize)) - track_ref.getTanlWord());
    } else {
      temp_eta = track_ref.getTanlWord();
    }
    Outword.eta = l1tmetemu::transformSignedValue(
        temp_eta, TTTrack_TrackWord::TrackBitWidths::kTanlSize, l1tmetemu::kInternalEtaWidth);

  } else {
    track_ref.setTrackWordBits();
    // Change track word digitization to digitization expected by track MET
    Outword.pt = l1tmetemu::digitizeSignedValue<TTTrack_TrackWord::rinv_t>(
        track_ref.momentum().perp(), l1tmetemu::kInternalPtWidth, l1tmetemu::kStepPt);

    Outword.eta = l1tmetemu::digitizeSignedValue<TTTrack_TrackWord::tanl_t>(
        abs(track_ref.momentum().eta()), l1tmetemu::kInternalEtaWidth, l1tmetemu::kStepEta);
  }

  Outword.chi2rphidof = track_ref.getChi2RPhiWord();
  Outword.chi2rzdof = track_ref.getChi2RZWord();
  Outword.bendChi2 = track_ref.getBendChi2Word();
  Outword.nstubs = countNStub(track_ref.getHitPatternWord());
  Outword.Hitpattern = track_ref.getHitPatternWord();
  Outword.Sector = track_ref.phiSector();
  Outword.EtaSector = (track_ref.getTanlWord() & (1 << (TTTrack_TrackWord::TrackBitWidths::kTanlSize - 1)));
  Outword.phi = track_ref.phi();
  Outword.globalPhi = localToGlobalPhi(track_ref.getPhiWord(), phiShift[track_ref.phiSector()]);

  unsigned int temp_pv = l1tmetemu::digitizeSignedValue<TTTrack_TrackWord::z0_t>(
      PV.z0(),
      TTTrack_TrackWord::TrackBitWidths::kZ0Size,
      TTTrack_TrackWord::stepZ0);  // Convert vertex to integer representation
  //Rescale to internal representations
  Outword.z0 = l1tmetemu::transformSignedValue(
      track_ref.getZ0Word(), TTTrack_TrackWord::TrackBitWidths::kZ0Size, l1tmetemu::kInternalVTXWidth);
  Outword.pV = l1tmetemu::transformSignedValue(
      temp_pv, TTTrack_TrackWord::TrackBitWidths::kZ0Size, l1tmetemu::kInternalVTXWidth);

  return Outword;
}

#endif
