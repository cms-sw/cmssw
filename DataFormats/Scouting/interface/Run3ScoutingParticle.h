#ifndef DataFormats_Scouting_Run3ScoutingParticle_h
#define DataFormats_Scouting_Run3ScoutingParticle_h

#include <vector>
#include <cstdint>

//class for holding PF candidate information, for use in data scouting
//IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class Run3ScoutingParticle {
public:
  //constructor with values for all data fields
  Run3ScoutingParticle(float pt,
                       float eta,
                       float phi,
                       int pdgId,
                       int vertex,
                       float normchi2,
                       float dz,
                       float dxy,
                       float dzsig,
                       float dxysig,
                       uint8_t lostInnerHits,
                       uint8_t quality,
                       float trk_pt,
                       float trk_eta,
                       float trk_phi,
                       bool relative_trk_vars)
      : pt_(pt),
        eta_(eta),
        phi_(phi),
        pdgId_(pdgId),
        vertex_(vertex),
        normchi2_(normchi2),
        dz_(dz),
        dxy_(dxy),
        dzsig_(dzsig),
        dxysig_(dxysig),
        lostInnerHits_(lostInnerHits),
        quality_(quality),
        trk_pt_(trk_pt),
        trk_eta_(trk_eta),
        trk_phi_(trk_phi),
        relative_trk_vars_(relative_trk_vars) {}

  // default constractor
  Run3ScoutingParticle()
      : pt_(0),
        eta_(0),
        phi_(0),
        pdgId_(0),
        vertex_(-1),
        normchi2_(0),
        dz_(0),
        dxy_(0),
        dzsig_(0),
        dxysig_(0),
        lostInnerHits_(0),
        quality_(0),
        trk_pt_(0),
        trk_eta_(0),
        trk_phi_(0),
        relative_trk_vars_(false) {}

  //accessor functions
  float pt() const { return pt_; }
  float eta() const { return eta_; }
  float phi() const { return phi_; }
  int pdgId() const { return pdgId_; }
  int vertex() const { return vertex_; }
  float normchi2() const { return normchi2_; }
  float dz() const { return dz_; }
  float dxy() const { return dxy_; }
  float dzsig() const { return dzsig_; }
  float dxysig() const { return dxysig_; }
  uint8_t lostInnerHits() const { return lostInnerHits_; }
  uint8_t quality() const { return quality_; }
  float trk_pt() const { return trk_pt_; }
  float trk_eta() const { return trk_eta_; }
  float trk_phi() const { return trk_phi_; }
  bool relative_trk_vars() const { return relative_trk_vars_; }

private:
  float pt_;
  float eta_;
  float phi_;
  int pdgId_;
  int vertex_;
  float normchi2_;
  float dz_;
  float dxy_;
  float dzsig_;
  float dxysig_;
  uint8_t lostInnerHits_;
  uint8_t quality_;
  float trk_pt_;
  float trk_eta_;
  float trk_phi_;
  bool relative_trk_vars_;
};

typedef std::vector<Run3ScoutingParticle> Run3ScoutingParticleCollection;

#endif  // DataFormats_Scouting_Run3ScoutingParticle_h
