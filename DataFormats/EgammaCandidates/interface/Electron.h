#ifndef EgammaCandidates_Electron_h
#define EgammaCandidates_Electron_h
/** \class reco::Electron
 *
 * Reco Candidates with an Electron component
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

namespace reco {

  class Electron : public RecoCandidate {
  public:
    /// default constructor
    Electron() : RecoCandidate() {}
    /// constructor from values
    Electron(Charge q, const LorentzVector& p4, const Point& vtx = Point(0, 0, 0))
        : RecoCandidate(q, p4, vtx, -11 * q) {}
    /// destructor
    ~Electron() override;
    /// returns a clone of the candidate
    Electron* clone() const override;
    /// reference to a Track
    using reco::RecoCandidate::track;  // avoid hiding the base
    reco::TrackRef track() const override;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster() const override;
    /// reference to a GsfTrack
    reco::GsfTrackRef gsfTrack() const override;
    /// set refrence to Photon component
    void setSuperCluster(const reco::SuperClusterRef& r) { superCluster_ = r; }
    /// set refrence to Track component
    void setTrack(const reco::TrackRef& r) { track_ = r; }
    /// set reference to GsfTrack component
    void setGsfTrack(const reco::GsfTrackRef& r) { gsfTrack_ = r; }

    bool isElectron() const override;

  private:
    /// check overlap with another candidate
    bool overlap(const Candidate&) const override;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    /// reference to a Track
    reco::TrackRef track_;
    /// reference to a GsfTrack;
    reco::GsfTrackRef gsfTrack_;
  };

}  // namespace reco

#endif
