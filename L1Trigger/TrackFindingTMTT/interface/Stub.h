#ifndef L1Trigger_TrackFindingTMTT_Stub_h
#define L1Trigger_TrackFindingTMTT_Stub_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/DigitalStub.h"
#include "L1Trigger/TrackFindingTMTT/interface/DegradeBend.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"

#include <vector>
#include <set>
#include <array>
#include <map>
#include <memory>

class TrackerGeometry;
class TrackerTopology;

namespace tmtt {

  class TP;
  class DegradeBend;
  class StubKiller;

  typedef edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> > TTStubDetSetVec;
  typedef edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_> > TTStubDetSet;
  typedef edm::Ref<TTStubDetSetVec, TTStub<Ref_Phase2TrackerDigi_> > TTStubRef;
  typedef edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_> >, TTCluster<Ref_Phase2TrackerDigi_> >
      TTClusterRef;
  typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_> TTStubAssMap;
  typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_> TTClusterAssMap;

  //=== Represents a Tracker stub (=pair of hits)

  class Stub {
  public:
    // Hybrid L1 tracking: stub constructor.
    Stub(const Settings* settings,
         unsigned int idStub,
         double phi,
         double r,
         double z,
         double bend,
         unsigned int iphi,
         double alpha,
         unsigned int layerId,
         unsigned int iPhiSec,
         bool psModule,
         bool barrel,
         bool tiltedBarrel,
         float stripPitch,
         float stripLength,
         unsigned int nStrips);

    // TMTT L1 tracking: stub constructor.
    Stub(const TTStubRef& ttStubRef,
         unsigned int index_in_vStubs,
         const Settings* settings,
         const TrackerTopology* trackerTopology,
         const TrackerModule* trackerModule,
         const DegradeBend* degradeBend,
         const StubKiller* stubKiller);

    bool operator==(const Stub& stubOther) { return (this->index() == stubOther.index()); }

    // Return reference to original TTStub.
    const TTStubRef& ttStubRef() const { return ttStubRef_; }

    // Info about tracker module containing stub.
    const TrackerModule* trackerModule() const { return trackerModule_; }

    // Fill truth info with association from stub to tracking particles.
    void fillTruth(const std::map<edm::Ptr<TrackingParticle>, const TP*>& translateTP,
                   const edm::Handle<TTStubAssMap>& mcTruthTTStubHandle,
                   const edm::Handle<TTClusterAssMap>& mcTruthTTClusterHandle);

    // Calculate HT m-bin range consistent with bend.
    void calcQoverPtrange();

    // Digitize stub for input to GP, HT, SF, TF
    enum class DigiStage { NONE, GP, HT, SF, TF };
    void digitize(unsigned int iPhiSec, DigiStage digiStep);

    // Control warning messages about accessing non-digitized quantities.
    void setDigitizeWarningsOn(bool newVal) { digitizeWarningsOn_ = newVal; }

    // Access to digitized version of stub coords.
    const DigitalStub* digitalStub() const { return digitalStub_.get(); }

    // === Functions for returning info about reconstructed stubs ===

    // Location in InputData::vStubs_
    unsigned int index() const { return index_in_vStubs_; }

    //--- Stub data and quantities derived from it ---

    // Stub coordinates (optionally after digitisation, if digitisation requested via cfg).
    // N.B. Digitisation is only run if Stub::digitize() is called.
    float phi() const { return phi_; }
    float r() const { return r_; }
    float z() const { return z_; }
    float theta() const { return atan2(r_, z_); }
    float eta() const { return asinh(z_ / r_); }

    // Location of stub in module in units of strip/pixel number in phi direction.
    // Range from 0 to (nStrips - 1) inclusive.
    unsigned int iphi() const { return iphi_; }
    // alpha correction for non-radial strips in endcap 2S modules.
    // (If true hit at larger r than stub r by deltaR, then stub phi needs correcting by +alpha*deltaR).
    // *** TO DO *** : Digitize this.
    float alpha() const { return alpha_; }

    // Get stub bend and its resolution, as available within the front end chip (i.e. prior to loss of bits
    // or digitisation).
    float bendInFrontend() const { return bendInFrontend_; }
    float bendCutInFrontend() const { return settings_->bendCut(); }
    // Get stub bend (i.e. displacement between two hits in stub in units of strip pitch).
    float bend() const { return bend_; }
    // Bend resolution.
    float bendCut() const { return (settings_->bendCut() + (numMergedBend_ - 1) * settings_->bendCutExtra()); }
    // No. of bend values merged into FE bend encoding of this stub.
    float numMergedBend() const { return numMergedBend_; }
    // Estimated track q/Pt based on stub bend info.
    float qOverPt() const { return (this->qOverPtOverBend() * this->bend()); }
    float qOverPtcut() const { return (this->qOverPtOverBend() * this->bendCut()); }
    // Range in q/Pt bins in HT array compatible with stub bend.
    unsigned int min_qOverPt_bin() const { return min_qOverPt_bin_; }
    unsigned int max_qOverPt_bin() const { return max_qOverPt_bin_; }
    // Difference in phi between stub and angle at which track crosses given radius, assuming track has given Pt.
    float phiDiff(float rad, float Pt) const { return std::abs(r_ - rad) * (settings_->invPtToDphi()) / Pt; }
    // Phi angle at which particle consistent with this stub & its bend cross specified radius.
    float trkPhiAtR(float rad) const { return phi_ + (bend_ * dphiOverBend_) * (1. - rad / r_); }
    // Its resolution
    float trkPhiAtRcut(float rad) const { return (bendCut() * dphiOverBend_) * std::abs(1. - rad / r_); }

    // -- conversion factors
    // Ratio of track crossing angle to bend.
    float dphiOverBend() const { return dphiOverBend_; }
    // Ratio of q/Pt to bend.
    float qOverPtOverBend() const { return dphiOverBend_ / (r_ * settings_->invPtToDphi()); }

    //--- Info about the two clusters that make up the stub.

    // Coordinates in frame of sensor, measured in units of strip pitch along two orthogonal axes running perpendicular and parallel to longer axis of pixels/strips (U & V).
    std::array<float, 2> localU_cluster() const { return localU_cluster_; }
    std::array<float, 2> localV_cluster() const { return localV_cluster_; }

    //--- Check if this stub will be output by FE. Stub failing this not used for L1 tracks.
    bool frontendPass() const { return frontendPass_; }
    // Indicates if stub would have passed DE cuts, were it not for window size encoded in DegradeBend.h
    bool stubFailedDegradeWindow() const { return stubFailedDegradeWindow_; }

    //--- Truth info

    // Association of stub to tracking particles
    const std::set<const TP*>& assocTPs() const {
      return assocTPs_;
    }  // Return TPs associated to this stub. (Whether only TPs contributing to both clusters are returned is determined by "StubMatchStrict" config param.)
    bool genuine() const { return (not assocTPs_.empty()); }  // Did stub match at least one TP?
    const TP* assocTP() const {
      return assocTP_;
    }  // If only one TP contributed to both clusters, this tells you which TP it is. Returns nullptr if none.

    // Association of both clusters making up stub to tracking particles
    std::array<bool, 2> genuineCluster() const {
      return std::array<bool, 2>{{(assocTPofCluster_[0] != nullptr), (assocTPofCluster_[1] != nullptr)}};
    }  // Was cluster produced by a single TP?
    std::array<const TP*, 2> assocTPofCluster() const {
      return assocTPofCluster_;
    }  // Which TP made each cluster. Warning: If cluster was not produced by a single TP, then returns nullptr! (P.S. If both clusters match same TP, then this will equal assocTP()).

    //--- Quantities common to all stubs in a given module ---
    // N.B. Not taken from trackerModule_ to cope with Hybrid tracking.

    // Angle between normal to module and beam-line along +ve z axis. (In range -PI/2 to +PI/2).
    float tiltAngle() const { return tiltAngle_; }
    // Uncertainty in stub (r,z)
    float sigmaR() const { return (barrel() ? 0. : sigmaPar()); }
    float sigmaZ() const { return (barrel() ? sigmaPar() : 0.); }
    // Hit resolution perpendicular to strip. Measures phi.
    float sigmaPerp() const { return invRoot12 * stripPitch_; }
    // Hit resolution parallel to strip. Measures r or z.
    float sigmaPar() const { return invRoot12 * stripLength_; }

    //--- These module variables could be taken directly from trackerModule_, were it not for need
    //--- to support Hybrid.
    // Module type: PS or 2S?
    bool psModule() const { return psModule_; }
    // Tracker layer ID number (1-6 = barrel layer; 11-15 = endcap A disk; 21-25 = endcap B disk)
    unsigned int layerId() const { return layerId_; }
    // Reduced layer ID (in range 1-7). This encodes the layer ID in only 3 bits (to simplify firmware) by merging some barrel layer and endcap disk layer IDs into a single ID.
    unsigned int layerIdReduced() const { return layerIdReduced_; }
    bool barrel() const { return barrel_; }
    // True if stub is in tilted barrel module.
    bool tiltedBarrel() const { return tiltedBarrel_; }
    // Strip pitch (or pixel pitch along shortest axis).
    float stripPitch() const { return stripPitch_; }
    // Strip length (or pixel pitch along longest axis).
    float stripLength() const { return stripLength_; }
    // No. of strips in sensor.
    unsigned int nStrips() const { return nStrips_; }

  private:
    // Degrade assumed stub bend resolution.
    // And return an integer indicating how many values of bend are merged into this single one.
    void degradeResolution(float bend, float& degradedBend, unsigned int& num) const;

    // Set the frontendPass_ flag, indicating if frontend readout electronics will output this stub.
    void setFrontend(const StubKiller* stubKiller);

    // Set info about the module that this stub is in.
    void setTrackerModule(const TrackerGeometry* trackerGeometry,
                          const TrackerTopology* trackerTopology,
                          const DetId& detId);

    // Function to calculate approximation for dphiOverBendCorrection aka B
    double approxB();

    // Calculate variables giving ratio of track intercept angle to stub bend.
    void calcDphiOverBend();

  private:
    TTStubRef ttStubRef_;  // Reference to original TTStub

    const Settings* settings_;  // configuration parameters.

    unsigned int index_in_vStubs_;  // location of this stub in InputData::vStubs

    //--- Parameters passed along optical links from PP to MP (or equivalent ones if easier for analysis software to use).
    // WARNING: If you add any variables in this section, take care to ensure that they are digitized correctly by Stub::digitize().
    float phi_;  // stub coords, optionally after digitisation.
    float r_;
    float z_;
    float bend_;                    // bend of stub.
    float dphiOverBend_;            // related to rho parameter.
    unsigned int min_qOverPt_bin_;  // Range in q/Pt bins in HT array compatible with stub bend.
    unsigned int max_qOverPt_bin_;

    //--- Info about the two clusters that make up the stub.
    std::array<float, 2> localU_cluster_;
    std::array<float, 2> localV_cluster_;

    unsigned int iphi_;
    float alpha_;

    // Would front-end electronics output this stub?
    bool frontendPass_;
    // Did stub fail window cuts assumed in DegradeBend.h?
    bool stubFailedDegradeWindow_;
    // Bend in front end chip (prior to degredation by loss of bits & digitization).
    float bendInFrontend_;
    // Used for stub bend resolution degrading.
    unsigned int numMergedBend_;

    //--- Truth info about stub.
    const TP* assocTP_;
    std::set<const TP*> assocTPs_;
    //--- Truth info about the two clusters that make up the stub
    std::array<const TP*, 2> assocTPofCluster_;

    std::unique_ptr<DigitalStub> digitalStub_;  // Class used to digitize stub if required.
    DigiStage lastDigiStep_;
    bool digitizeWarningsOn_;  // Enable warnings about accessing non-digitized quantities.

    // Info about tracker module containing stub.
    const TrackerModule* trackerModule_;

    // Used to degrade stub bend information.
    const DegradeBend* degradeBend_;

    // These module variables are needed only to support the Hybrid stub constructor.
    // (Otherwise, they could be taken from trackerModule_).
    unsigned int layerId_;
    unsigned int layerIdReduced_;
    float tiltAngle_;
    float stripPitch_;
    float stripLength_;
    unsigned int nStrips_;
    bool psModule_;
    bool barrel_;
    bool tiltedBarrel_;

    const float rejectedStubBend_ = 99999.;  // Bend set to this if stub rejected.

    const float invRoot12 = sqrt(1. / 12.);
  };

}  // namespace tmtt
#endif
