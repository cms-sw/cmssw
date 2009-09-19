#ifndef PhysicsTools_PatUtils_interface_JetIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_JetIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/Selector.h"

class JetIDSelectionFunctor : public Selector<pat::Jet>  {

 public: // interface

  enum Version_t { CRAFT08, N_VERSIONS };
  enum Quality_t { LOOSE, TIGHT, N_QUALITY};
  

 JetIDSelectionFunctor( Version_t version, Quality_t quality ) :
  version_(version), quality_(quality)
  {
    push_back("LOOSE"); 
    push_back("LOOSE_fHPD");
    push_back("LOOSE_N90Hits");
    push_back("LOOSE_EMF");

    push_back("TIGHT");
    push_back("TIGHT_fHPD");
    push_back("TIGHT_EMF");
    
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
    // LOOSE cuts    
    if ( this->operator[]("LOOSE") ) {

      if( this->operator[]("LOOSE_fHPD")    && jet.fHPD() >= 0.98 ) return false; // fHPD Cut
      if( this->operator[]("LOOSE_N90Hits") && jet.n90Hits() <= 1 ) return false; // N90Hits Cut
      double abs_eta = TMath::Abs( jet.eta() );
      double corrPt = jet.correctedP4( pat::JetCorrFactors::L3 ).Pt();

      // EMF Cut
      if ( this->operator[]("LOOSE_EMF") ) {
	if( abs_eta <= 2.55 ) { // HBHE
	  if( jet.emEnergyFraction() <= 0.01 ) return false;
	} else {                // HF
	  if( jet.emEnergyFraction() <= -0.9 ) return false;
	  if( corrPt > 80 && jet.emEnergyFraction() >= 1 ) return false;
	}
      }
 
      // TIGHT cuts
      if( this->operator[]("TIGHT") && quality_ == TIGHT ) {
	// jtf1hpd<0.98 && (jtrawpt<25 || jtf1hpd<0.95) && jthn90>1.5 
	//  && (abs(jtdeta)>2.55 || jtemf>0.01) && (jtpt<80 || abs(jtdeta)<1 || jtemf<1)
	if( this->operator[]("TIGHT_fHPD") && jet.pt() >= 25 && jet.fHPD() >= 0.95 ) return false;
	
	if ( this->operator[]("TIGHT_EMF") ) {
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
	      
	    } // end if HF
	  }// end if outside HBHE
	}// end if tight EMF cuts are set
      }// end if tight cuts are set
    }// end if loose cuts are set
    
    return true;

  }
  
 private: // member variables
  
  Version_t version_;
  Quality_t quality_;
  
};

#endif
