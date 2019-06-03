#ifndef MuonReco_MuonQuality_h
#define MuonReco_MuonQuality_h

#include "DataFormats/Math/interface/Point3D.h"
namespace reco {
  struct MuonQuality {
    ///
    /// bool returns true if standAloneMuon_updatedAtVtx was used in the fit
    bool updatedSta;
    /// value of the kink algorithm applied to the inner track stub
    float trkKink;
    /// value of the kink algorithm applied to the global track
    float glbKink;
    /// chi2 value for the inner track stub with respect to the global track
    float trkRelChi2;
    /// chi2 value for the outer track stub with respect to the global track
    float staRelChi2;
    /// chi2 value for the STA-TK matching of local position
    float chi2LocalPosition;
    /// chi2 value for the STA-TK matching of local momentum
    float chi2LocalMomentum;
    /// local distance seperation for STA-TK TSOS matching on same surface
    float localDistance;
    /// global delta-Eta-Phi of STA-TK matching
    float globalDeltaEtaPhi;
    /// if the STA-TK matching passed the tighter matching criteria
    bool tightMatch;
    /// the tail probability (-ln(P)) of the global fit
    float glbTrackProbability;

    /// Kink position for the tracker stub and global track
    math::XYZPoint tkKink_position;
    math::XYZPoint glbKink_position;

    MuonQuality()
        : updatedSta(false),
          trkKink(0),
          glbKink(0),
          trkRelChi2(0),
          staRelChi2(0),
          chi2LocalPosition(0),
          chi2LocalMomentum(0),
          localDistance(0),
          globalDeltaEtaPhi(0),
          tightMatch(false),
          glbTrackProbability(0) {}
  };
}  // namespace reco
#endif
