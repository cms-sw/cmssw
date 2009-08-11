#ifndef PhysicsTools_PatUtils_interface_JetIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_JetIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <functional>

class JetIDSelectionFunctor : public std::unary_function<pat::Jet, bool>  {

 public: // interface

  enum Version_t { CRAFT08, N_VERSIONS };
  enum Quality_t { LOOSE, TIGHT, N_QUALITY};


 JetIDSelectionFunctor( Version_t version, Quality_t quality ) :
  version_(version), quality_(quality)
  {
  }

  // Allow for multiple definitions of the cuts. 
  bool operator()( const pat::Jet & jet ) const 
  { 

    if ( version_ == CRAFT08 ) return craft08Cuts( jet );
    else {
      return false;
    }
  }

  // cuts based on craft 08 analysis. 
  bool craft08Cuts( const pat::Jet & jet) const
  {
    // Loose cuts
    if( jet.fHPD() >= 0.98 ) return false;
    if( jet.n90Hits() <= 1 ) return false;
    double abs_eta = TMath::Abs( jet.eta() );
    double corrPt = jet.correctedP4( pat::JetCorrFactors::L3 ).Pt();

    if( abs_eta <= 2.55 ) { // HBHE
      if( jet.emEnergyFraction() <= 0.01 ) return false;
    } else {                // HF
      if( jet.emEnergyFraction() <= -0.9 ) return false;
      if( corrPt > 80 && jet.emEnergyFraction() >= 1 ) return false;
    }
 
    if( quality_ == TIGHT ) {
      // jtf1hpd<0.98 && (jtrawpt<25 || jtf1hpd<0.95) && jthn90>1.5 
      //  && (abs(jtdeta)>2.55 || jtemf>0.01) && (jtpt<80 || abs(jtdeta)<1 || jtemf<1)
      if( jet.pt() >= 25 && jet.fHPD() >= 0.95 ) return false;

      if( abs_eta >= 1 && corrPt >= 80 && jet.emEnergyFraction() >= 1 ) return false; // outside HB

      if( abs_eta >= 2.55 ) { // outside HBHE

	if( jet.emEnergyFraction() <= -0.3 ) return false;

	if( abs_eta < 3.25 ) { // HE-HF transition region
	  // (abs(jtdeta)<2.55 || abs(jtdeta)>3.25 || 
	  // (jtemf>-0.3 && (jtpt<50 || jtemf>-0.2) && (jtpt<80 || jtemf>-0.1) && (jtpt<340 || jtemf<0.95)) )
	  if( corrPt >= 50 && jet.emEnergyFraction() <= -0.2 ) return false;
	  if( corrPt >= 80 && jet.emEnergyFraction() <= -0.1 ) return false;

	  if( corrPt >= 340 && jet.emEnergyFraction() >= 0.95 ) return false;

	} else { // HF
	  // (abs(jtdeta)<3.25 || (jtemf>-0.3 && jtemf<0.9 && (jtpt<50 || (jtemf>-0.2 &&
	  // jtemf<0.8)) && (jtpt<130 || (jtemf>-0.1 && jtemf<0.7))))
	  if( jet.emEnergyFraction() >= 0.9 ) return false;
	  
	  if( corrPt >= 50 && jet.emEnergyFraction() <= -0.2 ) return false;
	  if( corrPt >= 50 && jet.emEnergyFraction() >= 0.8 ) return false;
	  if( corrPt >= 130 && jet.emEnergyFraction() <= -0.1 ) return false;
	  if( corrPt >= 130 && jet.emEnergyFraction() >= 0.7 ) return false;
	  
	}
      }
      
    }

    return true;

  }
  
 private: // member variables
  
  Version_t version_;
  Quality_t quality_;
  
};

#endif
