#ifndef JetReco_GenericJet_h
#define JetReco_GenericJet_h

 /** \class reco::Jet
 *
 * \short Base class for all types of Jets
 *
 * GenericJet describes jets made from arbitrary constituents,
 * No direct constituents reference is stored for now 
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   Mar 23, 2007 by F.R.
 * \version   $Id: Jet.h,v 1.11 2006/12/11 12:21:39 fedor Exp $
 ************************************************************/
#include <string>
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/JetReco/interface/GenericJetfwd.h"

namespace reco {
  class GenericJet : public CompositeRefBaseCandidate {
  public:
    /// Default constructor
    GenericJet () {}
    /// Initiator
    GenericJet (const LorentzVector& fP4, const Point& fVertex, const std::vector<unsigned>& fConstituents);
    /// Destructor
    virtual ~GenericJet () {}

    /// # of constituents
    virtual int nConstituents () const;

    /// list of constituents
    std::vector <unsigned> getJetConstituents () const;

  /// Print object
    virtual std::string print () const;

  private:
    std::vector<unsigned> mConstituents;
  };
}
#endif
