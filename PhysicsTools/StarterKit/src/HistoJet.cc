#include "PhysicsTools/StarterKit/interface/HistoJet.h"


#include <iostream>

using pat::HistoJet;
using namespace std;

// Constructor:

HistoJet::HistoJet(  std::string dir, std::string group,std::string pre,
		     double pt1, double pt2, double m1, double m2 ) 
  : HistoGroup<Jet>( dir, group, pre, pt1, pt2, m1, m2)
{
  // book relevant jet histograms
  addHisto( h_jetFlavour_   =
	    new PhysVarHisto( pre + "Flavour", "Jet Flavour", 21, 0, 21, currDir_, "", "vD" )
	    );
  addHisto( h_BDiscriminant_=
	    new PhysVarHisto( pre + "BDiscriminant", "Jet B Discriminant", 100, -10, 90, currDir_, "", "vD")
	    );
  addHisto( h_jetCharge_    =
	    new PhysVarHisto( pre + "Charge", "Jet Charge", 100, -5, 5, currDir_, "", "vD")
	    );
  addHisto( h_nTrk_         =
	    new PhysVarHisto( pre + "NTrk", "Jet N_{TRK}", 51, -0.5, 50.5, currDir_, "", "vD" )
	    );
}

HistoJet::~HistoJet()
{
}


void HistoJet::fill( const Jet * jet, uint iJet, double weight )
{

  // First fill common 4-vector histograms
  HistoGroup<Jet>::fill( jet, iJet, weight );

  // fill relevant jet histograms
  h_jetFlavour_     ->fill( jet->partonFlavour(), iJet, weight );
  h_BDiscriminant_  ->fill( jet->bDiscriminator("trackCountingHighPurJetTags"), iJet, weight );
  h_jetCharge_      ->fill( jet->jetCharge(), iJet, weight );
  h_nTrk_           ->fill( jet->associatedTracks().size(), iJet, weight );

}

void HistoJet::fill( const reco::ShallowCloneCandidate * pshallow, uint iJet, double weight )
{

  // Get the underlying object that the shallow clone represents
  const pat::Jet * jet = dynamic_cast<const pat::Jet*>(pshallow);

  if ( jet == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a jet" << endl;
    return;
  }

  // First fill common 4-vector histograms from shallow clone
  HistoGroup<Jet>::fill( pshallow, iJet, weight);

  // fill relevant jet histograms
  h_jetFlavour_     ->fill( jet->partonFlavour(), iJet, weight );
  h_BDiscriminant_  ->fill( jet->bDiscriminator("trackCountingHighPurJetTags"), iJet, weight );
  h_jetCharge_      ->fill( jet->jetCharge(), iJet, weight );
  h_nTrk_           ->fill( jet->associatedTracks().size(), iJet, weight );

}


void HistoJet::fillCollection( const std::vector<Jet> & coll, double weight ) 
{
 
  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<Jet>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoJet::clearVec()
{
  HistoGroup<Jet>::clearVec();

  h_jetFlavour_     ->clearVec( );
  h_BDiscriminant_  ->clearVec( );
  h_jetCharge_      ->clearVec( );
  h_nTrk_           ->clearVec( );

}
