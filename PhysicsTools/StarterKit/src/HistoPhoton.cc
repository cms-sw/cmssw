#include "PhysicsTools/StarterKit/interface/HistoPhoton.h"

using namespace std;

// Constructor:

using pat::HistoPhoton;

HistoPhoton::HistoPhoton( std::string dir,
			      double pt1, double pt2, double m1, double m2 ) :
  HistoGroup<Photon>( dir, "Photon", "photon", pt1, pt2, m1, m2)
{
  // book relevant photon histograms

  addHisto( h_trackIso_      =
	    new PhysVarHisto("photonTrackIso",       "Photon Track Isolation"    , 100, 0, 1, currDir_, "", "vD")
	    );
  addHisto( h_caloIso_       =
	    new PhysVarHisto("photonCaloIso",        "Photon Calo Isolation"     , 100, 0, 1, currDir_, "", "vD")
	    );
  addHisto( h_photonID_      =
	    new PhysVarHisto("photonPhotonID",       "Photon ID"                 , 100, 0, 1, currDir_, "", "vD")
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

void HistoPhoton::clearVec()
{
  HistoGroup<Photon>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
  h_photonID_->clearVec();
}
