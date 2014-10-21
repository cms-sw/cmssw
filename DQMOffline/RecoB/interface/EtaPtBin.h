#ifndef EtaPtBin_H
#define EtaPtBin_H

#include <string>

#include "DataFormats/JetReco/interface/Jet.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"

/** \class EtaPtBin
 *
 *  Decide if jet/parton lie within desired rapidity/pt range.
 *
 */

class EtaPtBin {

 public:

  EtaPtBin(const bool& etaActive_ , const double& etaMin_ , const double& etaMax_ ,
	    const bool& ptActive_  , const double& ptMin_  , const double& ptMax_ ) ;

  ~EtaPtBin () {} ;

  /// String describes rapidity/pt range.
  std::string getDescriptionString () const { return descriptionString ; } 

  /// method to build the string from other quantities
  /// (static for easy external use)
  static std::string buildDescriptionString 
  	( const bool& etaActive_ , const double& etaMin_ , const double& etaMax_ ,
	  const bool& ptActive_  , const double& ptMin_  , const double& ptMax_ ) ;  // pt
  
  
  /// Get rapidity/pt ranges and check whether rapidity/pt cuts are active.
  bool   getEtaActive () const { return etaActive ; }
  double getEtaMin    () const { return etaMin    ; }
  double getEtaMax    () const { return etaMax    ; }

  bool   getPtActive () const { return ptActive ; }
  double getPtMin    () const { return ptMin    ; }
  double getPtMax    () const { return ptMax    ; }


  /// Check if jet/parton are within rapidity/pt cuts.
  bool inBin(const double & eta , const double & pt) const;
  bool inBin(const reco::Jet & jet, const double jec) const;
//   bool inBin(const BTagMCTools::JetFlavour & jetFlavour) const;

 private:

  // definition of the bin

  bool   etaActive ; // should cuts be applied?
  double etaMin ;
  double etaMax ;
  
  bool   ptActive ; // should cuts be applied?
  double ptMin ;
  double ptMax ;
  

  // description string as built from bin definition
  std::string descriptionString ;
  
  
} ;


#endif
