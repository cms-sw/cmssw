#include "PhysicsTools/StarterKit/interface/HistoComposite.h"

#include <iostream>

using std::cout;
using std::endl;
using reco::CompositeCandidate;

using pat::HistoComposite;

HistoComposite::
HistoComposite( std::string dir, std::string candTitle, std::string candName)
  :
  HistoGroup<CompositeCandidate>( dir, candTitle, candName ),
  candName_(candName)
{

  histoMuon_     = new HistoMuon    ( dir );
  histoElectron_ = new HistoElectron( dir );
  histoJet_      = new HistoJet     ( dir );
  histoMET_      = new HistoMET     ( dir );
  histoParticle_ = new HistoParticle( dir );
}

HistoComposite::~HistoComposite()
{
  if ( histoMuon_    ) delete histoMuon_;    ;
  if ( histoElectron_) delete histoElectron_ ;
  if ( histoJet_     ) delete histoJet_;     ;
  if ( histoMET_     ) delete histoMET_;     ;
  if ( histoParticle_) delete histoParticle_ ;
}

void HistoComposite::fill( const reco::CompositeCandidate * cand )
{

  // Fill 4-vector information for candidate
  HistoGroup<CompositeCandidate>::fill( cand );

  int imu = 1;
  int iele = 1;
  int ijet = 1;
  int imet = 1;

  // Now fill information for daughters
  for (unsigned int i = 0; i < cand->numberOfDaughters(); i++ ) {
    const reco::Candidate * c = cand->daughter(i);

    if      ( dynamic_cast<const Muon*>    ( c ) != 0 ) {
      histoMuon_    ->fill( dynamic_cast<const Muon*>    ( c ), imu );
      imu++;
    }
    else if ( dynamic_cast<const Electron*>( c ) != 0 ) {
      histoElectron_->fill( dynamic_cast<const Electron*>( c ), iele );
      iele++;
    }
    else if ( dynamic_cast<const Jet*>     ( c ) != 0 ) {
      histoJet_     ->fill( dynamic_cast<const Jet*>     ( c ), ijet );
      ijet++;
    }
    else if ( dynamic_cast<const MET*>     ( c ) != 0 ) {
      histoMET_     ->fill( dynamic_cast<const MET*>     ( c ), imet );
      imet++;
    }
    else if ( dynamic_cast<const reco::CompositeCandidate*> ( c ) != 0 ) {
      this          ->fill( dynamic_cast<const reco::CompositeCandidate*> (c) );
    }
  }
}

//void HistoComposite::fill( const reco::CompositeCandidate * cand )
//{
//  fill ( cand );
//}
