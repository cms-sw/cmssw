#ifndef RecoCandidate_RecoCaloTowerCandidate_h
#define RecoCandidate_RecoCaloTowerCandidate_h
// $Id: RecoCaloTowerCandidate.h,v 1.2 2006/02/21 10:37:35 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoCaloTowerCandidate : public RecoCandidate {
  public:
    RecoCaloTowerCandidate() : RecoCandidate() { }
    RecoCaloTowerCandidate( Charge q , const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) :
      RecoCandidate( q, p4, vtx ) { }
    virtual ~RecoCaloTowerCandidate();
    virtual RecoCaloTowerCandidate * clone() const;
    void setCaloTower( const CaloTowerRef & r ) { caloTower_ = r; }

  private:
    virtual CaloTowerRef caloTower() const;
    CaloTowerRef caloTower_;
  };
  
}

#endif
