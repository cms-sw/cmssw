#ifndef DataFormats_Run3ScoutingElectron_h
#define DataFormats_Run3ScoutingElectron_h

#include <vector>
#include <cstdint>

// Class for holding electron information, for use in data scouting
// IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class Run3ScoutingElectron {
public:
  //constructor with values for all data fields
  Run3ScoutingElectron(float pt,
                       float eta,
                       float phi,
                       float m,
                       std::vector<float> trkd0,
                       std::vector<float> trkdz,
                       std::vector<float> trkpt,
                       std::vector<float> trketa,
                       std::vector<float> trkphi,
                       std::vector<float> trkchi2overndf,
                       float dEtaIn,
                       float dPhiIn,
                       float sigmaIetaIeta,
                       float hOverE,
                       float ooEMOop,
                       int missingHits,
                       std::vector<int> trkcharge,
                       float ecalIso,
                       float hcalIso,
                       float trackIso,
                       float r9,
                       float sMin,
                       float sMaj,
                       uint32_t seedId,
                       std::vector<float> energyMatrix,
                       std::vector<uint32_t> detIds,
                       std::vector<float> timingMatrix,
                       bool rechitZeroSuppression)
      : pt_(pt),
        eta_(eta),
        phi_(phi),
        m_(m),
        trkd0_(std::move(trkd0)),
        trkdz_(std::move(trkdz)),
        trkpt_(std::move(trkpt)),
        trketa_(std::move(trketa)),
        trkphi_(std::move(trkphi)),
        trkchi2overndf_(std::move(trkchi2overndf)),
        dEtaIn_(dEtaIn),
        dPhiIn_(dPhiIn),
        sigmaIetaIeta_(sigmaIetaIeta),
        hOverE_(hOverE),
        ooEMOop_(ooEMOop),
        missingHits_(missingHits),
        trkcharge_(std::move(trkcharge)),
        ecalIso_(ecalIso),
        hcalIso_(hcalIso),
        trackIso_(trackIso),
        r9_(r9),
        sMin_(sMin),
        sMaj_(sMaj),
        seedId_(seedId),
        energyMatrix_(std::move(energyMatrix)),
        detIds_(std::move(detIds)),
        timingMatrix_(std::move(timingMatrix)),
        rechitZeroSuppression_(rechitZeroSuppression) {}
  //default constructor
  Run3ScoutingElectron()
      : pt_(0),
        eta_(0),
        phi_(0),
        m_(0),
        trkd0_(0),
        trkdz_(0),
        trkpt_(0),
        trketa_(0),
        trkphi_(0),
        trkchi2overndf_(0),
        dEtaIn_(0),
        dPhiIn_(0),
        sigmaIetaIeta_(0),
        hOverE_(0),
        ooEMOop_(0),
        missingHits_(0),
        trkcharge_(0),
        ecalIso_(0),
        hcalIso_(0),
        trackIso_(0),
        r9_(0),
        sMin_(0),
        sMaj_(0),
        seedId_(0),
        rechitZeroSuppression_(false) {}

  //accessor functions
  float pt() const { return pt_; }
  float eta() const { return eta_; }
  float phi() const { return phi_; }
  float m() const { return m_; }
  std::vector<float> const& trkd0() const { return trkd0_; }
  std::vector<float> const& trkdz() const { return trkdz_; }
  std::vector<float> const& trkpt() const { return trkpt_; }
  std::vector<float> const& trketa() const { return trketa_; }
  std::vector<float> const& trkphi() const { return trkphi_; }
  std::vector<float> const& trkchi2overndf() const { return trkchi2overndf_; }
  float dEtaIn() const { return dEtaIn_; }
  float dPhiIn() const { return dPhiIn_; }
  float sigmaIetaIeta() const { return sigmaIetaIeta_; }
  float hOverE() const { return hOverE_; }
  float ooEMOop() const { return ooEMOop_; }
  int missingHits() const { return missingHits_; }
  std::vector<int> const& trkcharge() const { return trkcharge_; }
  float ecalIso() const { return ecalIso_; }
  float hcalIso() const { return hcalIso_; }
  float trackIso() const { return trackIso_; }
  float r9() const { return r9_; }
  float sMin() const { return sMin_; }
  float sMaj() const { return sMaj_; }
  uint32_t seedId() const { return seedId_; }
  std::vector<float> const& energyMatrix() const { return energyMatrix_; }
  std::vector<uint32_t> const& detIds() const { return detIds_; }
  std::vector<float> const& timingMatrix() const { return timingMatrix_; }
  bool rechitZeroSuppression() const { return rechitZeroSuppression_; }

private:
  float pt_;
  float eta_;
  float phi_;
  float m_;
  std::vector<float> trkd0_;
  std::vector<float> trkdz_;
  std::vector<float> trkpt_;
  std::vector<float> trketa_;
  std::vector<float> trkphi_;
  std::vector<float> trkchi2overndf_;
  float dEtaIn_;
  float dPhiIn_;
  float sigmaIetaIeta_;
  float hOverE_;
  float ooEMOop_;
  int missingHits_;
  std::vector<int> trkcharge_;
  float ecalIso_;
  float hcalIso_;
  float trackIso_;
  float r9_;
  float sMin_;
  float sMaj_;
  uint32_t seedId_;
  std::vector<float> energyMatrix_;
  std::vector<uint32_t> detIds_;
  std::vector<float> timingMatrix_;
  bool rechitZeroSuppression_;
};

typedef std::vector<Run3ScoutingElectron> Run3ScoutingElectronCollection;

#endif
