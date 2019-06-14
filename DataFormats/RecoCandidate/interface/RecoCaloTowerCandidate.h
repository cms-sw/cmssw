#ifndef RecoCandidate_RecoCaloTowerCandidate_h
#define RecoCandidate_RecoCaloTowerCandidate_h
/** \class reco::RecoCaloTowerCandidate
 *
 * Reco Candidates with a CaloTower component
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoCaloTowerCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoCaloTowerCandidate() : RecoCandidate() {}
    /// constructor from values
    RecoCaloTowerCandidate(Charge q, const LorentzVector& p4, const Point& vtx = Point(0, 0, 0))
        : RecoCandidate(q, p4, vtx) {}
    /// constructor from values
    RecoCaloTowerCandidate(Charge q, const PolarLorentzVector& p4, const Point& vtx = Point(0, 0, 0))
        : RecoCandidate(q, p4, vtx) {}
    /// destructor
    ~RecoCaloTowerCandidate() override;
    /// returns a clone of the candidate
    RecoCaloTowerCandidate* clone() const override;
    /// set CaloTower reference
    void setCaloTower(const CaloTowerRef& r) { caloTower_ = r; }
    /// reference to a CaloTower
    CaloTowerRef caloTower() const override;

  private:
    /// check overlap with another candidate
    bool overlap(const Candidate&) const override;
    /// reference to a CaloTower
    CaloTowerRef caloTower_;
  };

}  // namespace reco

#endif
