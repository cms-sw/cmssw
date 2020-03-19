
#ifndef DataFormats_BTauReco_IsolatedTauTagInfo_h
#define DataFormats_BTauReco_IsolatedTauTagInfo_h
//
// \class IsolatedTauTagInfo
// \short Extended object for the Tau Isolation algorithm.
// contains the result and the methods used in the ConeIsolation Algorithm, to create the
// object to be made persistent on RECO
//
// \author: Simone Gennai, based on ORCA class by S. Gennai and F. Moortgat
//

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

namespace reco {

  class IsolatedTauTagInfo : public JTATagInfo {
  public:
    //default constructor
    IsolatedTauTagInfo(void)
        : JTATagInfo(),
          selectedTracks_()

    {}

    IsolatedTauTagInfo(const TrackRefVector& tracks, const JetTracksAssociationRef& jtaRef)
        : JTATagInfo(jtaRef), selectedTracks_(tracks) {}

    //destructor
    ~IsolatedTauTagInfo() override {}

    //get the tracks from the jetTag
    const TrackRefVector allTracks() const { return tracks(); }

    //get the selected tracks used to computed the isolation
    const TrackRefVector selectedTracks() const { return selectedTracks_; }

    IsolatedTauTagInfo* clone() const override { return new IsolatedTauTagInfo(*this); }

    // default discriminator: returns the value of the discriminator of the jet tag, i.e. the one computed with the parameters taken from the cfg file
    float discriminator() const { return m_discriminator; }
    //set discriminator value
    void setDiscriminator(double discriminator) { m_discriminator = discriminator; }

    // methods to be used to recomputed the isolation with a new set of parameters
    float discriminator(
        float m_cone, float sig_cone, float iso_con, float pt_min_lt, float pt_min_tk, int nTracksIsoRing = 0) const;
    float discriminator(const math::XYZVector& myVector,
                        float m_cone,
                        float sig_cone,
                        float iso_con,
                        float pt_min_lt,
                        float pt_min_tk,
                        int nTracksIsoRing) const;
    // Used in case the PV is not considered
    float discriminator(float m_cone,
                        float sig_cone,
                        float iso_con,
                        float pt_min_lt,
                        float pt_min_tk,
                        int nTracksIsoRing,
                        float dz_lt) const;
    float discriminator(const math::XYZVector& myVector,
                        float m_cone,
                        float sig_cone,
                        float iso_con,
                        float pt_min_lt,
                        float pt_min_tk,
                        int nTracksIsoRing,
                        float dz_lt) const;

    // return all tracks in a cone of size "size" around a direction "direction"
    const TrackRefVector tracksInCone(const math::XYZVector& myVector, const float size, const float pt_min) const;
    const TrackRefVector tracksInCone(const math::XYZVector& myVector,
                                      const float size,
                                      const float pt_min,
                                      const float z_pv,
                                      const float dz_lt) const;

    // return the leading track in a given cone around the jet axis or a given direction
    void setLeadingTrack(const TrackRef);
    const TrackRef leadingSignalTrack() const;
    const TrackRef leadingSignalTrack(const float rm_cone, const float pt_min) const;
    const TrackRef leadingSignalTrack(const math::XYZVector& myVector, const float rm_cone, const float pt_min) const;

  private:
    double m_discriminator;
    TrackRefVector selectedTracks_;
    TrackRef leadTrack_;
  };

  DECLARE_EDM_REFS(IsolatedTauTagInfo)

}  // namespace reco

#endif  // DataFormats_BTauReco_IsolatedTauTagInfo_h
