#ifndef L1Trigger_TrackFindingTMTT_L1track3D_h
#define L1Trigger_TrackFindingTMTT_L1track3D_h

#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1trackBase.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Sector.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"

#include <vector>
#include <string>
#include <unordered_set>
#include <utility>

//=== L1 track candidate found in 3 dimensions.
//=== Gives access to all stubs on track and to its 3D helix parameters.
//=== Also calculates & gives access to associated truth particle (Tracking Particle) if any.

namespace tmtt {

  class L1track3D : public L1trackBase {
  public:
    // Seeding layers of tracklet pattern reco.
    enum TrackletSeedType { L1L2, L2L3, L3L4, L5L6, D1D2, D3D4, L1D1, L2D1, L3L4L2, L5L6L4, L2L3D1, D1D2L2, NONE };

  public:
    L1track3D(const Settings* settings,
              const std::vector<Stub*>& stubs,
              std::pair<unsigned int, unsigned int> cellLocationHT,
              std::pair<float, float> helixRphi,
              std::pair<float, float> helixRz,
              float helixD0,
              unsigned int iPhiSec,
              unsigned int iEtaReg,
              unsigned int optoLinkID,
              bool mergedHTcell)
        : L1trackBase(),
          settings_(settings),
          stubs_(stubs),
          stubsConst_(stubs_.begin(), stubs_.end()),
          cellLocationHT_(cellLocationHT),
          helixRphi_(helixRphi),
          helixRz_(helixRz),
          helixD0_(helixD0),
          iPhiSec_(iPhiSec),
          iEtaReg_(iEtaReg),
          optoLinkID_(optoLinkID),
          mergedHTcell_(mergedHTcell),
          seedLayerType_(TrackletSeedType::NONE),
          seedPS_(999) {
      nLayers_ = Utility::countLayers(settings, stubs_);  // Count tracker layers these stubs are in
      matchedTP_ = Utility::matchingTP(settings,
                                       stubs_,
                                       nMatchedLayers_,
                                       matchedStubs_);  // Find associated truth particle & calculate info about match.
    }

    // TMTT tracking: constructor

    L1track3D(const Settings* settings,
              const std::vector<Stub*>& stubs,
              std::pair<unsigned int, unsigned int> cellLocationHT,
              std::pair<float, float> helixRphi,
              std::pair<float, float> helixRz,
              unsigned int iPhiSec,
              unsigned int iEtaReg,
              unsigned int optoLinkID,
              bool mergedHTcell)
        : L1track3D(
              settings, stubs, cellLocationHT, helixRphi, helixRz, 0.0, iPhiSec, iEtaReg, optoLinkID, mergedHTcell) {}

    ~L1track3D() override = default;

    //--- Set/get optional info for tracklet tracks.

    // Tracklet seeding layer pair (from Tracklet::calcSeedIndex())
    void setSeedLayerType(unsigned int seedLayerType) { seedLayerType_ = static_cast<TrackletSeedType>(seedLayerType); }
    TrackletSeedType seedLayerType() const { return seedLayerType_; }

    // Tracklet seed stub pair uses PS modules (from FPGATracket::PSseed())
    void setSeedPS(unsigned int seedPS) { seedPS_ = seedPS; }
    unsigned int seedPS() const { return seedPS_; }

    // Best stub (stub with smallest Phi residual in each layer/disk)
    void setBestStubs(std::unordered_set<const Stub*> bestStubs) { bestStubs_ = bestStubs; }
    std::unordered_set<const Stub*> bestStubs() const { return bestStubs_; }

    //--- Get information about the reconstructed track.

    // Get stubs on track candidate.
    const std::vector<const Stub*>& stubsConst() const override { return stubsConst_; }
    const std::vector<Stub*>& stubs() const override { return stubs_; }
    // Get number of stubs on track candidate.
    unsigned int numStubs() const override { return stubs_.size(); }
    // Get number of tracker layers these stubs are in.
    unsigned int numLayers() const override { return nLayers_; }
    // Get cell location of track candidate in r-phi Hough Transform array in units of bin number.
    std::pair<unsigned int, unsigned int> cellLocationHT() const override { return cellLocationHT_; }
    // The two conventionally agreed track helix parameters relevant in r-phi plane. i.e. (q/Pt, phi0)
    std::pair<float, float> helixRphi() const { return helixRphi_; }
    // The two conventionally agreed track helix parameters relevant in r-z plane. i.e. (z0, tan_lambda)
    std::pair<float, float> helixRz() const { return helixRz_; }

    //--- Return chi variables, (both digitized & undigitized), which are the stub coords. relative to track.

    std::vector<float> chiPhi() {
      std::vector<float> result;
      for (const Stub* s : stubs_) {
        float chi_phi = reco::deltaPhi(s->phi(), this->phi0() - s->r() * this->qOverPt() * settings_->invPtToDphi());
        result.push_back(chi_phi);
      }
      return result;
    }

    std::vector<int> chiPhiDigi() {
      std::vector<int> result;
      const float phiMult = pow(2, settings_->phiSBits()) / settings_->phiSRange();
      for (const float& chi_phi : this->chiPhi()) {
        int iDigi_chi_phi = floor(chi_phi * phiMult);
        result.push_back(iDigi_chi_phi);
      }
      return result;
    }

    std::vector<float> chiZ() {
      std::vector<float> result;
      for (const Stub* s : stubs_) {
        float chi_z = s->z() - (this->z0() + s->r() * this->tanLambda());
        result.push_back(chi_z);
      }
      return result;
    }

    std::vector<int> chiZDigi() {
      std::vector<int> result;
      const float zMult = pow(2, settings_->zBits()) / settings_->zRange();
      for (const float& chi_z : this->chiZ()) {
        int iDigi_chi_z = floor(chi_z * zMult);
        result.push_back(iDigi_chi_z);
      }
      return result;
    }

    //--- User-friendlier access to the helix parameters.

    float qOverPt() const override { return helixRphi_.first; }
    float charge() const { return (this->qOverPt() > 0 ? 1 : -1); }
    float invPt() const { return std::abs(this->qOverPt()); }
    // Protect pt against 1/pt = 0.
    float pt() const {
      constexpr float small = 1.0e-6;
      return 1. / (small + this->invPt());
    }
    float d0() const { return helixD0_; }  // Hough transform assumes d0 = 0.
    float phi0() const override { return helixRphi_.second; }
    float z0() const { return helixRz_.first; }
    float tanLambda() const { return helixRz_.second; }
    float theta() const { return atan2(1., this->tanLambda()); }  // Use atan2 to ensure 0 < theta < pi.
    float eta() const { return -log(tan(0.5 * this->theta())); }

    // Phi and z coordinates at which track crosses "chosenR" values used by r-phi HT and rapidity sectors respectively.
    float phiAtChosenR() const {
      return reco::deltaPhi(this->phi0() - (settings_->invPtToDphi() * settings_->chosenRofPhi()) * this->qOverPt(),
                            0.);
    }
    float zAtChosenR() const {
      return (this->z0() + (settings_->chosenRofZ()) * this->tanLambda());
    }  // neglects transverse impact parameter & track curvature.

    //--- Get phi sector and eta region used by track finding code that this track is in.
    unsigned int iPhiSec() const override { return iPhiSec_; }
    unsigned int iEtaReg() const override { return iEtaReg_; }

    //--- Opto-link ID used to send this track from HT to Track Fitter
    unsigned int optoLinkID() const override { return optoLinkID_; }

    //--- Was this track produced from a marged HT cell (e.g. 2x2)?
    bool mergedHTcell() const { return mergedHTcell_; }

    //--- Get information about its association (if any) to a truth Tracking Particle.

    // Get best matching tracking particle (=nullptr if none).
    const TP* matchedTP() const override { return matchedTP_; }
    // Get the matched stubs with this Tracking Particle
    const std::vector<const Stub*>& matchedStubs() const override { return matchedStubs_; }
    // Get number of matched stubs with this Tracking Particle
    unsigned int numMatchedStubs() const override { return matchedStubs_.size(); }
    // Get number of tracker layers with matched stubs with this Tracking Particle
    unsigned int numMatchedLayers() const override { return nMatchedLayers_; }
    // Get purity of stubs on track candidate (i.e. fraction matching best Tracking Particle)
    float purity() const { return numMatchedStubs() / float(numStubs()); }

    //--- For debugging purposes.

    // Remove incorrect stubs from the track using truth information.
    // Also veto tracks where the HT cell estimated from the true helix parameters is inconsistent with the cell the HT found the track in, (since probable duplicates).
    // Also veto tracks that match a truth particle not used for the algo efficiency measurement.
    // Return a boolean indicating if the track should be kept. (i.e. Is genuine & non-duplicate).
    bool cheat() {
      bool keep = false;

      std::vector<Stub*> stubsSel;
      if (matchedTP_ != nullptr) {  // Genuine track
        for (Stub* s : stubs_) {
          const TP* tp = s->assocTP();
          if (tp != nullptr) {
            if (matchedTP_->index() == tp->index()) {
              stubsSel.push_back(s);  // This stub was produced by same truth particle as rest of track, so keep it.
            }
          }
        }
      }
      stubs_ = stubsSel;
      stubsConst_ = std::vector<const Stub*>(stubs_.begin(), stubs_.end());

      nLayers_ = Utility::countLayers(settings_, stubs_);  // Count tracker layers these stubs are in
      matchedTP_ = Utility::matchingTP(settings_,
                                       stubs_,
                                       nMatchedLayers_,
                                       matchedStubs_);  // Find associated truth particle & calculate info about match.

      bool genuine = (matchedTP_ != nullptr);

      if (genuine && matchedTP_->useForAlgEff()) {
        Sector secTmp(settings_, iPhiSec_, iEtaReg_);
        HTrphi htRphiTmp(settings_, iPhiSec_, iEtaReg_, secTmp.etaMin(), secTmp.etaMax(), secTmp.phiCentre());
        std::pair<unsigned int, unsigned int> trueCell = htRphiTmp.trueCell(matchedTP_);

        std::pair<unsigned int, unsigned int> htCell = this->cellLocationHT();
        bool consistent = (htCell == trueCell);  // If true, track is probably not a duplicate.
        if (mergedHTcell_) {
          // If this is a merged cell, check other elements of merged cell.
          std::pair<unsigned int, unsigned int> htCell10(htCell.first + 1, htCell.second);
          std::pair<unsigned int, unsigned int> htCell01(htCell.first, htCell.second + 1);
          std::pair<unsigned int, unsigned int> htCell11(htCell.first + 1, htCell.second + 1);
          if (htCell10 == trueCell)
            consistent = true;
          if (htCell01 == trueCell)
            consistent = true;
          if (htCell11 == trueCell)
            consistent = true;
        }
        if (consistent)
          keep = true;
      }

      return keep;  // Indicate if track should be kept.
    }

  private:
    //--- Configuration parameters
    const Settings* settings_;

    //--- Information about the reconstructed track.
    std::vector<Stub*> stubs_;
    std::vector<const Stub*> stubsConst_;
    std::unordered_set<const Stub*> bestStubs_;
    unsigned int nLayers_;
    std::pair<unsigned int, unsigned int> cellLocationHT_;
    std::pair<float, float> helixRphi_;
    std::pair<float, float> helixRz_;
    float helixD0_;
    unsigned int iPhiSec_;
    unsigned int iEtaReg_;
    unsigned int optoLinkID_;
    bool mergedHTcell_;

    //--- Optional info used for tracklet tracks.
    TrackletSeedType seedLayerType_;
    unsigned int seedPS_;

    //--- Information about its association (if any) to a truth Tracking Particle.
    const TP* matchedTP_;
    std::vector<const Stub*> matchedStubs_;
    unsigned int nMatchedLayers_;
  };

}  // namespace tmtt

#endif
