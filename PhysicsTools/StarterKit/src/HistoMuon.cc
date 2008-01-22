#include "PhysicsTools/StarterKit/interface/HistoMuon.h"

#include <iostream>
#include <sstream>

using pat::HistoMuon;
using namespace std;

// Constructor:


HistoMuon::HistoMuon(std::string dir)
  : HistoGroup<Muon>( dir, "Muon", "mu")
{
  addHisto( h_trackIso_ =
	    new PhysVarHisto( "muTrackIso", "Muon Track Isolation", 20, 0, 10, currDir_, "", "vD" )
	   );

  addHisto( h_caloIso_  =
	    new PhysVarHisto( "muCaloIso",  "Muon Calo Isolation",  20, 0, 10, currDir_, "", "vD" )
	    );

  addHisto( h_leptonID_ =
	    new PhysVarHisto( "muLeptonID", "Muon Lepton ID",       20, 0, 1, currDir_, "", "vD" )
	    );
}



void HistoMuon::fill( const Muon *muon, uint iMu )
{

  // First fill common 4-vector histograms

  HistoGroup<Muon>::fill( muon, iMu);

  // fill relevant muon histograms
  h_trackIso_->fill( muon->trackIso(), iMu );
  h_caloIso_ ->fill( muon->caloIso() , iMu );
  h_leptonID_->fill( muon->leptonID(), iMu );
}


void HistoMuon::clearVec()
{
  HistoGroup<Muon>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
  h_leptonID_->clearVec();
}
