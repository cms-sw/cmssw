#include "PhysicsTools/StarterKit/interface/HistoPhoton.h"

using namespace std;

// Constructor:

using pat::HistoPhoton;

HistoPhoton::HistoPhoton( std::string dir, std::string role,std::string pre,
			      double pt1, double pt2, double m1, double m2 ) :
  HistoGroup<Photon>( dir, role, pre, pt1, pt2, m1, m2)
{
  // book relevant photon histograms

  addHisto( h_trackIso_      =
	    new PhysVarHisto(pre + "TrackIso",       "Photon Track Isolation"    , 100, 0, 1, currDir_, "", "vD")
	    );
  addHisto( h_caloIso_       =
	    new PhysVarHisto(pre + "CaloIso",        "Photon Calo Isolation"     , 100, 0, 1, currDir_, "", "vD")
	    );
  addHisto( h_photonID_      =
	    new PhysVarHisto(pre + "PhotonID",       "Photon ID"                 , 100, 0, 1, currDir_, "", "vD")
	    );
}

HistoPhoton::~HistoPhoton()
{
}


void HistoPhoton::fill( const Photon * photon, uint iE )
{

  // First fill common 4-vector histograms
  HistoGroup<Photon>::fill( photon, iE );

  // fill relevant photon histograms
  h_trackIso_       ->fill( photon->trackIso(), iE );
  h_caloIso_        ->fill( photon->caloIso(), iE );
  h_photonID_       ->fill( photon->photonID(), iE );

}



void HistoPhoton::fill( const reco::ShallowCloneCandidate * pshallow, uint iE )
{


  // Get the underlying object that the shallow clone represents
  const pat::Photon * photon = dynamic_cast<const pat::Photon*>(pshallow);

  if ( photon == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a photon" << endl;
    return;
  }

  // First fill common 4-vector histograms
  HistoGroup<Photon>::fill( pshallow, iE );

  // fill relevant photon histograms
  h_trackIso_       ->fill( photon->trackIso(), iE );
  h_caloIso_        ->fill( photon->caloIso(), iE );
  h_photonID_       ->fill( photon->photonID(), iE );

}


void HistoPhoton::fillCollection( const std::vector<Photon> & coll ) 
{

   h_size_->fill( coll.size() );     //! Save the size of the collection.

  std::vector<Photon>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoPhoton::clearVec()
{
  HistoGroup<Photon>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
  h_photonID_->clearVec();
}
