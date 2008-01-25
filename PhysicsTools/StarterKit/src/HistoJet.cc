#include "PhysicsTools/StarterKit/interface/HistoJet.h"

using pat::HistoJet;

// Constructor:

HistoJet::HistoJet(  std::string dir,
		     double pt1, double pt2, double m1, double m2 ) 
  : HistoGroup<Jet>( dir, "Jet", "jet", pt1, pt2, m1, m2)
{
  // book relevant jet histograms
  addHisto( h_jetFlavour_   =
	    new PhysVarHisto("jetFlavour", "Jet Flavour", 21, 0, 21, currDir_, "", "vD" )
	    );
  addHisto( h_BDiscriminant_=
	    new PhysVarHisto("jetBDiscriminant", "Jet B Discriminant", 100, -10, 90, currDir_, "", "vD")
	    );
  addHisto( h_jetCharge_    =
	    new PhysVarHisto("jetCharge", "Jet Charge", 100, -5, 5, currDir_, "", "vD")
	    );
  addHisto( h_nTrk_         =
	    new PhysVarHisto("jetNTrk", "Jet N_{TRK}", 51, -0.5, 50.5, currDir_, "", "vD" )
	    );
}

HistoJet::~HistoJet()
{
}


void HistoJet::fill( const Jet * jet, uint iJet )
{

  // First fill common 4-vector histograms
  HistoGroup<Jet>::fill( jet, iJet );

  // fill relevant jet histograms
  h_jetFlavour_     ->fill( jet->partonFlavour(), iJet );
  h_BDiscriminant_  ->fill( jet->bDiscriminator("trackCountingHighPurJetTags"), iJet );
  h_jetCharge_      ->fill( jet->jetCharge(), iJet );
  h_nTrk_           ->fill( jet->associatedTracks().size(), iJet );

}

void HistoJet::clearVec()
{
  HistoGroup<Jet>::clearVec();

  h_jetFlavour_     ->clearVec( );
  h_BDiscriminant_  ->clearVec( );
  h_jetCharge_      ->clearVec( );
  h_nTrk_           ->clearVec( );

}
