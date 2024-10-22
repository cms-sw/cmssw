#ifndef ParticleFlowCandidate_PFCandidatePhotonExtra_h
#define ParticleFlowCandidate_PFCandidatePhotonExtra_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"

#include <iosfwd>

namespace reco {
  /** \class reco::PFCandidatePhotonExtra
 *
 * extra information on the photon particle candidate from particle flow
 *
 */
  class PFCandidatePhotonExtra {
  public:
    /// constructor
    PFCandidatePhotonExtra();
    /// constructor
    PFCandidatePhotonExtra(const reco::SuperClusterRef&);
    /// destructor
    ~PFCandidatePhotonExtra() { ; }

    // variables for the single conversion identification

    /// return a reference to the corresponding supercluster
    reco::SuperClusterRef superClusterRef() const { return scRef_; }

    /// add Single Leg Conversion TrackRef
    void addSingleLegConvTrackRef(const reco::TrackRef& trackref);

    /// return vector of Single Leg Conversion TrackRef from
    const std::vector<reco::TrackRef>& singleLegConvTrackRef() const { return assoSingleLegRefTrack_; }

    /// add Single Leg Conversion mva
    void addSingleLegConvMva(float& mvasingleleg);

    /// return Single Leg Conversion mva
    const std::vector<float>& singleLegConvMva() const { return assoSingleLegMva_; }

    /// add Conversions from PF
    void addConversionRef(const reco::ConversionRef& convref);

    /// return Conversions from PF
    reco::ConversionRefVector conversionRef() const { return assoConversionsRef_; }

    //from Mustache Id:
    void setMustache_Et(float Must_Et) { Mustache_Et_ = Must_Et; }
    void setExcludedClust(int excluded) { Excluded_clust_ = excluded; }
    float Mustache_Et() const { return Mustache_Et_; }
    int ExcludedClust() const { return Excluded_clust_; }

    //MVA Energy Regression:
    void setMVAGlobalCorrE(float GCorr) { GlobalCorr_ = GCorr; }
    float MVAGlobalCorrE() const { return GlobalCorr_; }

    void setMVAGlobalCorrEError(float GCorr) { GlobalCorrEError_ = GCorr; }
    float MVAGlobalCorrEError() const { return GlobalCorrEError_; }

    void addLCorrClusEnergy(float LCorrE);
    const std::vector<float>& LCorrClusEnergy() const { return LocalCorr_; }

    void SetPFPhotonRes(float Res) { MVAResolution_ = Res; }
    float PFPhotonRes() const { return MVAResolution_; }

  private:
    /// Ref to supercluster
    reco::SuperClusterRef scRef_;

    ///  vector of TrackRef from Single Leg conversions
    std::vector<reco::TrackRef> assoSingleLegRefTrack_;

    ///  vector of Mvas from Single Leg conversions
    std::vector<float> assoSingleLegMva_;

    /// vector of ConversionRef from PF
    reco::ConversionRefVector assoConversionsRef_;

    //for Mustache_Id
    float Mustache_Et_;
    int Excluded_clust_;

    //for MVA Regression Energy
    std::vector<float> LocalCorr_;
    float GlobalCorr_;
    float GlobalCorrEError_;
    float MVAResolution_;
  };
}  // namespace reco
#endif
