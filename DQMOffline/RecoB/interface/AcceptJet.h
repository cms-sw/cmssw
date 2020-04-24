#ifndef AcceptJet_H
#define AcceptJet_H

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/Common/interface/Handle.h"

/** \class AcceptJet
 *
 *  Decide if jet and associated parton satisfy desired kinematic cuts.
 *
 */

class AcceptJet {

 public:
  AcceptJet(const double& etaMin_, const double& etaMax_, const double& ptMin_, const double& ptMax_,
            const double& pMin_, const double& pMax_, const double& ratioMin_, const double& ratioMax_,
	    const bool& doJetID_);
  /// Returns true if jet and associated parton satisfy kinematic cuts.
  bool operator() (const reco::Jet & jet, const int & jetFlavour, const edm::Handle<reco::SoftLeptonTagInfoCollection> & infos, const double jec) const;

  /// Set cut parameters
  void setEtaMin            ( double d ) { etaMin            = d ; } 
  void setEtaMax            ( double d ) { etaMax            = d ; } 
//   void setPPartonMin        ( double d ) { pPartonMin        = d ; } 
//   void setPPartonMax        ( double d ) { pPartonMax        = d ; } 
  void setPtRecJetMin       ( double d ) { ptRecJetMin       = d ; } 
  void setPtRecJetMax       ( double d ) { ptRecJetMax       = d ; } 
  void setPRecJetMin        ( double d ) { pRecJetMin        = d ; } 
  void setPRecJetMax        ( double d ) { pRecJetMax        = d ; }
  void setRatioMin          ( double d ) { ratioMin          = d ; }
  void setRatioMax          ( double d ) { ratioMax          = d ; }
  void setDoJetID           ( bool   b ) { doJetID           = b ; }

//   void setPtPartonMin       ( double d ) { ptPartonMin       = d ; } 
//   void setPtPartonMax       ( double d ) { ptPartonMax       = d ; } 

 protected:

  /// Finds the ratio of the momentum of any leptons in the jet to jet energy
  double ratio(const reco::Jet & jet, const edm::Handle<reco::SoftLeptonTagInfoCollection> & infos) const;

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

  double ratioMin ;
  double ratioMax ;
  
  //Apply loose Jet ID in case of PF jets
  bool doJetID;
} ;

#endif
