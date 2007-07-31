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
 * \version   $Id: BasicJet.h,v 1.5 2007/05/03 21:13:17 fedor Exp $
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
#endif
