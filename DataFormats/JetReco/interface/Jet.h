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
 * \version   $Id: Jet.h,v 1.7 2006/07/21 19:23:14 fedor Exp $
 ************************************************************/
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h"

namespace reco {
  class Jet : public CompositeRefCandidate {
  public:
    /// Default constructor
    Jet () : mNumberOfConstituents (0) {}
    /// Initiator
    Jet (const LorentzVector& fP4, const Point& fVertex, int fNumberOfConstituents)
      :  CompositeRefCandidate (0, fP4, fVertex),
      mNumberOfConstituents (fNumberOfConstituents) {}
    /// Destructor
    virtual ~Jet () {}
    
  protected:
    void setNConstituents (int fNConstituents) {mNumberOfConstituents = fNConstituents;}
  private:
    /** Number of constituents of the Jet*/
    int mNumberOfConstituents;
  };
}
#endif
