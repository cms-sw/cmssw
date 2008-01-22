#include "PhysicsTools/StarterKit/interface/HistoElectron.h"

using namespace std;

// Constructor:

using pat::HistoElectron;

HistoElectron::HistoElectron( std::string dir ) :
  HistoGroup<Electron>( dir, "Electron", "e")
{
  // book relevant electron histograms

  addHisto( h_trackIso_      =
	    new PhysVarHisto("eTrackIso",       "Electron Track Isolation"    , 100, 0, 1, currDir_, "", "vD")
	    );
  addHisto( h_caloIso_       =
	    new PhysVarHisto("eCaloIso",        "Electron Calo Isolation"     , 100, 0, 1, currDir_, "", "vD")
	    );
  addHisto( h_leptonID_      =
	    new PhysVarHisto("eLeptonID",       "Electron Lepton ID"          , 100, 0, 1, currDir_, "", "vD")
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

void HistoElectron::clearVec()
{
  HistoGroup<Electron>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
  h_leptonID_->clearVec();
}
