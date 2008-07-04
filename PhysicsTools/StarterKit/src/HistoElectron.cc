#include "PhysicsTools/StarterKit/interface/HistoElectron.h"

using namespace std;

// Constructor:

using pat::HistoElectron;

HistoElectron::HistoElectron( std::string dir,std::string group,std::string pre,
			      double pt1, double pt2, double m1, double m2 ) :
  HistoGroup<Electron>( dir, group, pre, pt1, pt2, m1, m2)
{
  // book relevant electron histograms

  addHisto( h_trackIso_      =
	    new PhysVarHisto( pre + "TrackIso",       "Electron Track Isolation"    , 100, 0, 10, currDir_, "", "vD")
	    );
  addHisto( h_caloIso_       =
	    new PhysVarHisto( pre + "CaloIso",        "Electron Calo Isolation"     , 100, -20, 20, currDir_, "", "vD")
	    );
  addHisto( h_leptonID_      =
	    new PhysVarHisto( pre + "LeptonID",       "Electron Lepton ID"          , 100, 0, 1, currDir_, "", "vD")
	    );
}

HistoElectron::~HistoElectron()
{
}


void HistoElectron::fill( const Electron * electron, uint iE, double weight )
{

  // First fill common 4-vector histograms
  HistoGroup<Electron>::fill( electron, iE , weight);

  // fill relevant electron histograms
  h_trackIso_       ->fill( electron->trackIso(), iE, weight );
  h_caloIso_        ->fill( electron->caloIso(), iE, weight );
  h_leptonID_       ->fill( electron->leptonID(), iE, weight );

}


void HistoElectron::fill( const reco::ShallowClonePtrCandidate * pshallow, uint iE, double weight )
{

  // Get the underlying object that the shallow clone represents
  const pat::Electron * electron = dynamic_cast<const pat::Electron*>(pshallow);

  if ( electron == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a electron" << endl;
    return;
  }


  // First fill common 4-vector histograms
  HistoGroup<Electron>::fill( pshallow, iE, weight );

  // fill relevant electron histograms
  h_trackIso_       ->fill( electron->trackIso(), iE, weight );
  h_caloIso_        ->fill( electron->caloIso(), iE, weight );
  h_leptonID_       ->fill( electron->leptonID(), iE, weight );

}


void HistoElectron::fillCollection( const std::vector<Electron> & coll,double weight ) 
{
 
  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<Electron>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoElectron::clearVec()
{
  HistoGroup<Electron>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
  h_leptonID_->clearVec();
}
