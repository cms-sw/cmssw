#ifndef RecoCandidate_RecoCaloJetCandidate_h
#define RecoCandidate_RecoCaloJetCandidate_h
/** \class reco::RecoCaloJetCandidate
 *
 * Reco Candidates with a Track component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoCaloJetCandidate.h,v 1.3 2006/04/26 07:56:20 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoCaloJetCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoCaloJetCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoCaloJetCandidate( Charge q , const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) :
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~RecoCaloJetCandidate();
    /// returns a clone of the candidate
    virtual RecoCaloJetCandidate * clone() const;
    /// set reference to caloJet
    void setCaloJet( const CaloJetRef & r ) { caloJet_ = r; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a caloJet
    virtual CaloJetRef caloJet() const;
    /// reference to a caloJet
    CaloJetRef caloJet_;
  };
  
}

#endif
