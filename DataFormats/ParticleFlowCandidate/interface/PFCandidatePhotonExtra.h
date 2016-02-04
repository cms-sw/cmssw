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
    ~PFCandidatePhotonExtra(){;}

    // variables for the single conversion identification

    /// return a reference to the corresponding supercluster
    reco::SuperClusterRef superClusterRef() const {return scRef_ ; }

    /// add Single Leg Conversion TrackRef 
    void addSingleLegConvTrackRef(const reco::TrackRef& trackref);

    /// return vector of Single Leg Conversion TrackRef from 
    const std::vector<reco::TrackRef>& singleLegConvTrackRef() const {return assoSingleLegRefTrack_;}

    /// add Single Leg Conversion mva
    void addSingleLegConvMva(float& mvasingleleg);

    /// return Single Leg Conversion mva
    const std::vector<float>& singleLegConvMva() const {return assoSingleLegMva_;}

    /// add Conversions from PF
    void addConversionRef(const reco::ConversionRef& convref);

    /// return Conversions from PF
    reco::ConversionRefVector conversionRef() const {return assoConversionsRef_;} 


 private:
    
    /// Ref to supercluster
    reco::SuperClusterRef scRef_;

    ///  vector of TrackRef from Single Leg conversions
    std::vector<reco::TrackRef> assoSingleLegRefTrack_;

    ///  vector of Mvas from Single Leg conversions
    std::vector<float> assoSingleLegMva_;

    /// vector of ConversionRef from PF
    reco::ConversionRefVector assoConversionsRef_;

  };
}
#endif
