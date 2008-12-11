#ifndef AcceptJet_H
#define AcceptJet_H

#include "DataFormats/JetReco/interface/Jet.h"

/** \class AcceptJet
 *
 *  Decide if jet and associated parton satisfy desired kinematic cuts.
 *
 */

class AcceptJet {

 public:
  AcceptJet();
  /// Returns true if jet and associated parton satisfy kinematic cuts.
  bool operator() (const reco::Jet & jet, const int & jetFlavour) const;

  /// Set cut parameters
  void setEtaMin            ( double d ) { etaMin            = d ; } 
  void setEtaMax            ( double d ) { etaMax            = d ; } 
//   void setPPartonMin        ( double d ) { pPartonMin        = d ; } 
//   void setPPartonMax        ( double d ) { pPartonMax        = d ; } 
  void setPtRecJetMin       ( double d ) { ptRecJetMin       = d ; } 
  void setPtRecJetMax       ( double d ) { ptRecJetMax       = d ; } 
  void setPRecJetMin        ( double d ) { pRecJetMin        = d ; } 
  void setPRecJetMax        ( double d ) { pRecJetMax        = d ; } 

//   void setPtPartonMin       ( double d ) { ptPartonMin       = d ; } 
//   void setPtPartonMax       ( double d ) { ptPartonMax       = d ; } 

 protected:

  // eta range 
  double etaMin ;   // these are meant as |eta| !!
  double etaMax ;

  // parton p
//   double pPartonMin ;
//   double pPartonMax ;

  // parton pt
//   double ptPartonMin ;
//   double ptPartonMax ;

  // rec. jet
  double ptRecJetMin ;
  double ptRecJetMax ;
  //
  double pRecJetMin  ;
  double pRecJetMax  ;
  
  
} ;

#endif
