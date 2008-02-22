#include "PhysicsTools/StarterKit/interface/HistoTau.h"

#include <iostream>
#include <sstream>

using pat::HistoTau;
using namespace std;

// Constructor:


HistoTau::HistoTau(std::string dir,
		   double pt1, double pt2, double m1, double m2)
  : HistoGroup<Tau>( dir, "Tau", "tau", pt1, pt2, m1, m2)
{

  addHisto( h_nSignalTracks_ = 
	    new PhysVarHisto( "tauNSignalTracks", "Tau Number of Tracks in Signal Cone", 20, 0, 20, currDir_, "", "vD")
	    );

  addHisto( h_nIsolationTracks_ = 
	    new PhysVarHisto( "tauNIsolationTracks", "Tau Number of Tracks in Isolation Cone", 20, 0, 20, currDir_, "", "vD")
	    );

  addHisto( h_emEnergyFraction_ =
	    new PhysVarHisto( "tauEmEnergyFraction", "Tau EM Energy Fraction", 20, 0, 10, currDir_, "", "vD" )
	   );

  addHisto( h_eOverP_  =
	    new PhysVarHisto( "tauEOverP",  "Tau E over P",  20, 0, 10, currDir_, "", "vD" )
	    );


}



void HistoTau::fill( const Tau *tau, uint iMu )
{

  // First fill common 4-vector histograms

  HistoGroup<Tau>::fill( tau, iMu);

  // fill relevant tau histograms
  h_nSignalTracks_->fill( tau->signalTracks().size() );
  h_nIsolationTracks_->fill( tau->isolationTracks().size() );
  h_emEnergyFraction_->fill( tau->emEnergyFraction(), iMu );
  h_eOverP_ ->fill( tau->eOverP() , iMu );
}


void HistoTau::clearVec()
{
  HistoGroup<Tau>::clearVec();

  h_emEnergyFraction_->clearVec();
  h_eOverP_->clearVec();
}
