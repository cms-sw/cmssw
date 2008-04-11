#include "PhysicsTools/StarterKit/interface/HistoComposite.h"

#include <iostream>

using namespace std;

using namespace pat;

HistoComposite::
HistoComposite( std::string dir, std::string candTitle, std::string candName,
		double pt1, double pt2, double m1, double m2)
  :
  HistoGroup<NamedCompositeCandidate>( dir, candTitle, candName, pt1, pt2, m1, m2 ),
  candName_(candName)
{
}

HistoComposite::~HistoComposite()
{
  if ( histoMuon_.map.size() > 0 ) {
    HistoMap<HistoMuon>::map_type::iterator begin = histoMuon_.map.begin();
    HistoMap<HistoMuon>::map_type::iterator end = histoMuon_.map.end();
    for ( HistoMap<HistoMuon>::map_type::iterator i = begin;
	  i != end; ++i ) if ( i->second ) delete i->second ;
  }

  if ( histoElectron_.map.size() > 0 ) {
    HistoMap<HistoElectron>::map_type::iterator begin = histoElectron_.map.begin();
    HistoMap<HistoElectron>::map_type::iterator end = histoElectron_.map.end();
    for ( HistoMap<HistoElectron>::map_type::iterator i = begin;
	  i != end; ++i ) if ( i->second ) delete i->second ;
  }

  if ( histoTau_.map.size() > 0 ) {
    HistoMap<HistoTau>::map_type::iterator begin = histoTau_.map.begin();
    HistoMap<HistoTau>::map_type::iterator end = histoTau_.map.end();
    for ( HistoMap<HistoTau>::map_type::iterator i = begin;
	  i != end; ++i ) if ( i->second ) delete i->second ;
  }

  if ( histoJet_.map.size() > 0 ) {
    HistoMap<HistoJet>::map_type::iterator begin = histoJet_.map.begin();
    HistoMap<HistoJet>::map_type::iterator end = histoJet_.map.end();
    for ( HistoMap<HistoJet>::map_type::iterator i = begin;
	  i != end; ++i ) if ( i->second ) delete i->second ;
  }

  if ( histoMET_.map.size() > 0 ) {
    HistoMap<HistoMET>::map_type::iterator begin = histoMET_.map.begin();
    HistoMap<HistoMET>::map_type::iterator end = histoMET_.map.end();
    for ( HistoMap<HistoMET>::map_type::iterator i = begin;
	  i != end; ++i ) if ( i->second ) delete i->second ;
  }

  if ( histoPhoton_.map.size() > 0 ) {
    HistoMap<HistoPhoton>::map_type::iterator begin = histoPhoton_.map.begin();
    HistoMap<HistoPhoton>::map_type::iterator end = histoPhoton_.map.end();
    for ( HistoMap<HistoPhoton>::map_type::iterator i = begin;
	  i != end; ++i ) if ( i->second ) delete i->second ;
  }

  if ( histoTrack_.map.size() > 0 ) {
    HistoMap<HistoTrack>::map_type::iterator begin = histoTrack_.map.begin();
    HistoMap<HistoTrack>::map_type::iterator end = histoTrack_.map.end();
    for ( HistoMap<HistoTrack>::map_type::iterator i = begin;
	  i != end; ++i ) if ( i->second ) delete i->second ;
  }

  if ( histoComposite_.map.size() > 0 ) {
    HistoMap<HistoComposite>::map_type::iterator begin = histoComposite_.map.begin();
    HistoMap<HistoComposite>::map_type::iterator end = histoComposite_.map.end();
    for ( HistoMap<HistoComposite>::map_type::iterator i = begin;
	  i != end; ++i ) if ( i->second ) delete i->second ;
  }
}

void HistoComposite::fill( const reco::NamedCompositeCandidate * cand )
{


  // Fill 4-vector information for candidate
  HistoGroup<NamedCompositeCandidate>::fill( cand );

  const vector<string> & roles = cand->roles();

  if ( roles.size() != cand->numberOfDaughters() ) {
    cout << "HistoComposite::fill: Error: Nroles should match Ndaughters" << endl;
    return;
  }


  // Now fill information for daughters
  for (unsigned int i = 0; i < cand->numberOfDaughters(); ++i ) {
//     cout << "processing component " << i << endl;
    const reco::Candidate * c = cand->daughter(i);
    string role = roles[i];

//     cout << "Role = " << roles[i] << endl;
//     cout << "c = " << c << endl;
//     cout << "pdgid = " << c->pdgId() << endl;
//     cout << "pt = " << c->pt() << endl;


    const Muon *       pcmuon  = dynamic_cast<const Muon*>( c );
    const Electron *   pcelectron = dynamic_cast<const Electron*>( c );
    const Tau *        pctau = dynamic_cast<const Tau*>( c );
    const Jet *        pcjet = dynamic_cast<const Jet*>( c );
    const MET *        pcmet = dynamic_cast<const MET*>( c );
    const Photon *     pcphoton = dynamic_cast<const Photon *>( c );
    const reco::RecoChargedCandidate *    pctrack = dynamic_cast<const reco::RecoChargedCandidate*>(c);
    const reco::NamedCompositeCandidate * pccomposite = dynamic_cast<const reco::NamedCompositeCandidate*>(c);

    // The pointers might be in shallow clones, so check for that too

    if ( c->hasMasterClone() ) {
      if ( pcmuon == 0 )       pcmuon  = dynamic_cast<const Muon*>( &*(c->masterClone()) );
      if ( pcelectron == 0 )   pcelectron = dynamic_cast<const Electron*>( &*(c->masterClone()) );
      if ( pctau == 0 )        pctau = dynamic_cast<const Tau*>( &*(c->masterClone()) );
      if ( pcjet == 0 )        pcjet = dynamic_cast<const Jet*>( &*(c->masterClone()) );
      if ( pcmet == 0 )        pcmet = dynamic_cast<const MET*>( &*(c->masterClone()) );
      if ( pcphoton == 0 )     pcphoton = dynamic_cast<const Photon *>( &*(c->masterClone()) );
      if ( pctrack == 0 )      pctrack = dynamic_cast<const reco::RecoChargedCandidate*>( &*(c->masterClone()) );
      if ( pccomposite == 0 )  pccomposite = dynamic_cast<const reco::NamedCompositeCandidate*>( &*(c->masterClone()) );
    }


    if      ( pcmuon != 0 ) {
//       cout << "Filling muon" << endl;
      if ( histoMuon_.map.find( role ) == histoMuon_.map.end() ) {
	histoMuon_.map[role] = new HistoMuon( dir_, role, role ) ;
      }
      histoMuon_.map[role]    ->fill( pcmuon );
    }

    if      ( pcelectron != 0 ) {
//       cout << "Filling electron" << endl;
      if ( histoElectron_.map.find( role ) == histoElectron_.map.end() ) {
	histoElectron_.map[role] = new HistoElectron( dir_, role, role ) ;
      }
      histoElectron_.map[role]    ->fill( pcelectron );
    }

    if      ( pctau != 0 ) {
//       cout << "Filling tau" << endl;
      if ( histoTau_.map.find( role ) == histoTau_.map.end() ) {
	histoTau_.map[role] = new HistoTau( dir_, role, role ) ;
      }
      histoTau_.map[role]    ->fill( pctau );
    }


    if      ( pcjet != 0 ) {
//       cout << "Filling jet" << endl;
      if ( histoJet_.map.find( role ) == histoJet_.map.end() ) {
// 	cout << "Making histojet" << endl;
	histoJet_.map[role] = new HistoJet( dir_, role, role ) ;
// 	cout << "done" << endl;
      }
//       cout << "About to fill histojet" << endl;
      histoJet_.map[role]    ->fill( pcjet );
//       cout << "done" << endl;
    }


    if      ( pcmet != 0 ) {
//       cout << "Filling MET" << endl;
      if ( histoMET_.map.find( role ) == histoMET_.map.end() ) {
	histoMET_.map[role] = new HistoMET( dir_, role, role ) ;
      }
      histoMET_.map[role]    ->fill( pcmet );
    }


    if      ( pcphoton != 0 ) {
//       cout << "Filling photon" << endl;
      if ( histoPhoton_.map.find( role ) == histoPhoton_.map.end() ) {
	histoPhoton_.map[role] = new HistoPhoton( dir_, role, role ) ;
      }
      histoPhoton_.map[role]    ->fill( pcphoton );
    }


    if      ( pctrack != 0 ) {
//       cout << "Filling track" << endl;
      if ( histoTrack_.map.find( role ) == histoTrack_.map.end() ) {
	histoTrack_.map[role] = new HistoTrack( dir_, role, role ) ;
      }
      histoTrack_.map[role]    ->fill( pctrack );
    }


    if      ( pccomposite != 0 ) {
//       cout << "Filling composite with name " << pccomposite->name() << endl;
      if ( histoComposite_.map.find( role ) == histoComposite_.map.end() ) {
	histoComposite_.map[role] = new HistoComposite( dir_, role, role ) ;
      }
      histoComposite_.map[role]    ->fill( pccomposite );
    }

  }
}
