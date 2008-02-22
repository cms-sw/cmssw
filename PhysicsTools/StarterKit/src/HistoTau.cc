#include "PhysicsTools/StarterKit/interface/HistoTau.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <iostream>
#include <sstream>

using pat::HistoTau;
using namespace std;

// Constructor:


HistoTau::HistoTau(std::string dir,
		   double pt1, double pt2, double m1, double m2)
  : HistoGroup<Tau>( dir, "Tau", "tau", pt1, pt2, m1, m2)
{

  histoSignalTrack_ = new HistoTrack( dir, "TauSignalTracks", "tauSignalTracks" );
  histoIsolationTrack_ = new HistoTrack( dir, "TauIsolationTracks", "tauIsolationTracks" );

  addHisto( h_emEnergyFraction_ =
	    new PhysVarHisto( "tauEmEnergyFraction", "Tau EM Energy Fraction", 20, 0, 10, currDir_, "", "vD" )
	   );

  addHisto( h_eOverP_  =
	    new PhysVarHisto( "tauEOverP",  "Tau E over P",  20, 0, 10, currDir_, "", "vD" )
	    );


}



void HistoTau::fill( const Tau *tau, uint iTau )
{

  // First fill common 4-vector histograms

  HistoGroup<Tau>::fill( tau, iTau);

  // fill relevant tau histograms

  const double M_PION = 0.13957018;

  for ( unsigned int isignal = 0; isignal < tau->signalTracks().size(); isignal++ ) {
    const reco::Track & trk = *( tau->signalTracks().at(isignal) );
    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > p4;
    p4.SetPt( trk.pt() );
    p4.SetEta( trk.eta() );
    p4.SetPhi( trk.phi() );
    p4.SetM( M_PION );
    reco::Particle::LorentzVector p4_2( p4.x(), p4.y(), p4.z(), p4.t() );
    reco::RecoChargedCandidate trk_p4( trk.charge(), p4_2 );
    histoSignalTrack_->fill( &trk_p4, isignal );
  }
  for ( unsigned int iisolation = 0; iisolation < tau->isolationTracks().size(); iisolation++ ) {
    const reco::Track & trk = *( tau->isolationTracks().at(iisolation) );
    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > p4;
    p4.SetPt( trk.pt() );
    p4.SetEta( trk.eta() );
    p4.SetPhi( trk.phi() );
    p4.SetM( M_PION );
    reco::Particle::LorentzVector p4_2( p4.x(), p4.y(), p4.z(), p4.t() );
    reco::RecoChargedCandidate trk_p4( trk.charge(), p4_2 );
    histoIsolationTrack_->fill( &trk_p4, iisolation );
  }

  h_emEnergyFraction_->fill( tau->emEnergyFraction(), iTau );
  h_eOverP_ ->fill( tau->eOverP() , iTau );
}


void HistoTau::clearVec()
{
  HistoGroup<Tau>::clearVec();

  h_emEnergyFraction_->clearVec();
  h_eOverP_->clearVec();
}
