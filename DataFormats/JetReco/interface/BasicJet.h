#ifndef JetReco_BasicJet_h
#define JetReco_BasicJet_h

/** \class reco::BasicJet
 *
 * \short Jets made from CaloTowers
 *
 * BasicJet represents generic Jets witjout
 * any specific information
 * in addition to generic Jet parameters
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   $Id: BasicJet.h,v 1.6 2007/07/31 02:19:00 fedor Exp $
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"

namespace reco {
class BasicJet : public Jet {
 public:
  
  /** Default constructor*/
  BasicJet() {}
  
  /** Constructor from values*/
  BasicJet(const LorentzVector& fP4, const Point& fVertex, const Jet::Constituents& fConstituents);
  
  virtual ~BasicJet() {};

  /// Polymorphic clone
  virtual BasicJet* clone () const;

  /// Print object
  virtual std::string print () const;
  
 private:
  /// Polymorphic overlap
  virtual bool overlap( const Candidate & ) const;
};
}
#include "DataFormats/JetReco/interface/BasicJetCollection.h" // temporary fix before include_checcker runs globally
#endif
