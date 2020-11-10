#ifndef DataFormats_ScoutingPhoton_h
#define DataFormats_ScoutingPhoton_h

#include <vector>

// Class for holding photon information, for use in data scouting
// IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class ScoutingPhoton {
public:
  //constructor with values for all data fields
  ScoutingPhoton(float pt,
                 float eta,
                 float phi,
                 float m,
                 float sigmaIetaIeta,
                 float hOverE,
                 float ecalIso,
                 float hcalIso,
                 float trkIso,
                 float r9,
                 float sMin,
                 float sMaj,
                 std::vector<float> energyMatrix,
                 std::vector<float> timingMatrix)
      : pt_(pt),
        eta_(eta),
        phi_(phi),
        m_(m),
        sigmaIetaIeta_(sigmaIetaIeta),
        hOverE_(hOverE),
        ecalIso_(ecalIso),
        hcalIso_(hcalIso),
        trkIso_(trkIso),
        r9_(r9),
        sMin_(sMin),
        sMaj_(sMaj),
        energyMatrix_(std::move(energyMatrix)),
        timingMatrix_(std::move(timingMatrix)) {}
  //default constructor
  ScoutingPhoton()
      : pt_(0),
        eta_(0),
        phi_(0),
        m_(0),
        sigmaIetaIeta_(0),
        hOverE_(0),
        ecalIso_(0),
        hcalIso_(0),
        trkIso_(0),
        r9_(0),
        sMin_(0),
        sMaj_(0),
        energyMatrix_(0),
        timingMatrix_(0) {}

  //accessor functions
  float pt() const { return pt_; }
  float eta() const { return eta_; }
  float phi() const { return phi_; }
  float m() const { return m_; }
  float sigmaIetaIeta() const { return sigmaIetaIeta_; }
  float hOverE() const { return hOverE_; }
  float ecalIso() const { return ecalIso_; }
  float hcalIso() const { return hcalIso_; }
  float trkIso() const { return trkIso_; }
  float r9() const { return r9_; }
  float sMin() const { return sMin_; }
  float sMaj() const { return sMaj_; }
  std::vector<float> energyMatrix() const { return energyMatrix_; }
  std::vector<float> timingMatrix() const { return timingMatrix_; }

private:
  float pt_;
  float eta_;
  float phi_;
  float m_;
  float sigmaIetaIeta_;
  float hOverE_;
  float ecalIso_;
  float hcalIso_;
  float trkIso_;
  float r9_;
  float sMin_;
  float sMaj_;
  std::vector<float> energyMatrix_;
  std::vector<float> timingMatrix_;
};

typedef std::vector<ScoutingPhoton> ScoutingPhotonCollection;

#endif
