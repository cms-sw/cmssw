#ifndef RecoCandidate_RecoStandAloneMuonCandidate_h
#define RecoCandidate_RecoStandAloneMuonCandidate_h
/** \class reco::RecoStandAloneMuonCandidate
 *
 * Reco Candidates with a Track component
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoStandAloneMuonCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoStandAloneMuonCandidate() : RecoCandidate() {}
    /// constructor from values
    RecoStandAloneMuonCandidate(
        Charge q, const LorentzVector& p4, const Point& vtx = Point(0, 0, 0), int pdgId = 0, int status = 0)
        : RecoCandidate(q, p4, vtx, pdgId, status) {}
    /// constructor from values
    RecoStandAloneMuonCandidate(
        Charge q, const PolarLorentzVector& p4, const Point& vtx = Point(0, 0, 0), int pdgId = 0, int status = 0)
        : RecoCandidate(q, p4, vtx, pdgId, status) {}
    /// destructor
    ~RecoStandAloneMuonCandidate() override;
    /// returns a clone of the candidate
    RecoStandAloneMuonCandidate* clone() const override;
    /// set reference to track
    void setTrack(const reco::TrackRef& r) { standAloneMuonTrack_ = r; }
    /// reference to a track
    reco::TrackRef standAloneMuon() const override;

  private:
    /// check overlap with another candidate
    bool overlap(const Candidate&) const override;
    /// reference to a track
    reco::TrackRef standAloneMuonTrack_;
  };

}  // namespace reco

#endif
