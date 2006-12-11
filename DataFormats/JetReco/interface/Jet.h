#ifndef JetReco_Jet_h
#define JetReco_Jet_h

 /** \class reco::Jet
 *
 * \short Base class for all types of Jets
 *
 * Jet describes properties common for all kinds of jets, 
 * essentially kinematics. Base class for all types of Jets.
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   Original: April 22, 2005 by Fernando Varela Rodriguez.
 * \version   May 23, 2006 by F.R.
 * \version   $Id: Jet.h,v 1.10 2006/12/08 21:15:11 fedor Exp $
 ************************************************************/
#include <string>
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h"

namespace reco {
  class Jet : public CompositeRefCandidate {
  public:
    typedef std::vector<reco::CandidateRef> Constituents;
    /// Default constructor
    Jet () {}
    /// Initiator
    Jet (const LorentzVector& fP4, const Point& fVertex, const Constituents& fConstituents);
    /// Destructor
    virtual ~Jet () {}

    /// # of constituents
    virtual int nConstituents () const {return numberOfDaughters();}

    /// list of constituents
    Constituents getJetConstituents () const;

  /// Print object
    virtual std::string print () const;

  private:
    // disallow constituents modifications
    void addDaughter( const CandidateRef & fRef) {CompositeRefCandidate::addDaughter (fRef);}
  };
}
#endif
