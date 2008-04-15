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


void HistoElectron::fill( const Electron * electron, uint iE )
{

  // First fill common 4-vector histograms
  HistoGroup<Electron>::fill( electron, iE );

  // fill relevant electron histograms
  h_trackIso_       ->fill( electron->trackIso(), iE );
  h_caloIso_        ->fill( electron->caloIso(), iE );
  h_leptonID_       ->fill( electron->leptonID(), iE );

}


void HistoElectron::fillCollection( const std::vector<Electron> & coll ) 
{
 
  h_size_->fill( coll.size() );     //! Save the size of the collection.

  std::vector<Electron>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoElectron::clearVec()
{
  HistoGroup<Electron>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
  h_leptonID_->clearVec();
}
