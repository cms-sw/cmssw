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
 * \version   $Id: GenericJet.h,v 1.4 2007/07/31 02:19:01 fedor Exp $
 ************************************************************/
#include <string>
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"

namespace reco {
  class GenericJet : public CompositeRefBaseCandidate {
  public:
    /// Default constructor
    GenericJet () {}
    /// Initiator
    GenericJet (const LorentzVector& fP4, const Point& fVertex, const std::vector<CandidateBaseRef>& fConstituents);
    /// Destructor
    virtual ~GenericJet () {}

    /// # of constituents
    virtual int nConstituents () const;

  /// Print object
    virtual std::string print () const;

  };
}
#include "DataFormats/JetReco/interface/GenericJetCollection.h" // temporary fix before include_checcker runs globally
#endif
