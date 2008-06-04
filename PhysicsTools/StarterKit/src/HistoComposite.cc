#include "PhysicsTools/StarterKit/interface/HistoComposite.h"

#include <iostream>

using namespace std;
using namespace reco;
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
//     cout << "-------------processing component " << i << endl;
    const reco::Candidate * c = cand->daughter(i);
    string role = roles[i];

//     cout << "Role = " << roles[i] << endl;
//     cout << "pdgid = " << c->pdgId() << endl;
//     cout << "pt = " << c->pt() << endl;

    // Figure out what the candidate is based on type
    const Muon *       pcmuon  = dynamic_cast<const Muon*>( c );
    const Electron *   pcelectron = dynamic_cast<const Electron*>( c );
    const Tau *        pctau = dynamic_cast<const Tau*>( c );
    const Jet *        pcjet = dynamic_cast<const Jet*>( c );
    const MET *        pcmet = dynamic_cast<const MET*>( c );
    const Photon *     pcphoton = dynamic_cast<const Photon *>( c );
    const reco::RecoChargedCandidate *    pctrack = dynamic_cast<const reco::RecoChargedCandidate*>(c);
    const reco::NamedCompositeCandidate * pccomposite = dynamic_cast<const reco::NamedCompositeCandidate*>(c);

    // The pointers might be in shallow clones, so check for that too
    const reco::ShallowCloneCandidate * pshallow = dynamic_cast<const reco::ShallowCloneCandidate *>(c);
    if ( pccomposite == 0 && c->hasMasterClone() )  pccomposite = dynamic_cast<const reco::NamedCompositeCandidate*>( &*(c->masterClone()) );

    // The pointers might be in a ref, so check for that too
    const MuonRef    * prmuon     = dynamic_cast<const MuonRef*>    ( c );
    const ElectronRef* prelectron = dynamic_cast<const ElectronRef*>( c );
    const TauRef     * prtau      = dynamic_cast<const TauRef*>     ( c );
    const PhotonRef  * prphoton   = dynamic_cast<const PhotonRef*>  ( c );
    const JetRef     * prjet      = dynamic_cast<const JetRef*>     ( c );
    const METRef     * prmet      = dynamic_cast<const METRef*>     ( c );
    // Pick out the ref's kinematics and ignore the candidate's kinematics 
    // (why are they not the same???)
    if ( prmuon != 0 )       pcmuon     = &(*(prmuon->ref()));
    if ( prelectron != 0 )   pcelectron = &(*(prelectron->ref()));
    if ( prtau != 0 )        pctau      = &(*(prtau->ref()));
    if ( prjet != 0 )        pcjet      = &(*(prjet->ref()));
    if ( prmet != 0 )        pcmet      = &(*(prmet->ref()));
    if ( prphoton != 0 )     pcphoton   = &(*(prphoton->ref()));

    // ------------------------------------------------------
    // Fill histograms if the candidate is a muon
    // ------------------------------------------------------
    if      ( pcmuon != 0 ) {
//        cout << "Filling muon" << endl;
       // Here is where we do not yet have a histogram for this role
      if ( histoMuon_.map.find( role ) == histoMuon_.map.end() ) {
	histoMuon_.map[role] = new HistoMuon( dir_, role, role ) ;
      }
      // Here is if the candidate is a shallow clone, we need to
      // fill kinematic information from the shallow clone and 
      // detector information from the base object
      if ( c->hasMasterClone() ) {
	histoMuon_.map[role]    ->fill( pshallow );
      }
      // Here is if the candidate is a straightforward pointer
      else {
	histoMuon_.map[role]    ->fill( pcmuon );
      }
    }

    // ------------------------------------------------------
    // Fill histograms if the candidate is a electron
    // ------------------------------------------------------
    if      ( pcelectron != 0 ) {
//        cout << "Filling electron" << endl;
       // Here is where we do not yet have a histogram for this role
      if ( histoElectron_.map.find( role ) == histoElectron_.map.end() ) {
	histoElectron_.map[role] = new HistoElectron( dir_, role, role ) ;
      }
      // Here is if the candidate is a shallow clone, we need to
      // fill kinematic information from the shallow clone and 
      // detector information from the base object
      if ( c->hasMasterClone() ) {
	histoElectron_.map[role]    ->fill( pshallow );
      }
      // Here is if the candidate is a straightforward pointer
      else {
	histoElectron_.map[role]    ->fill( pcelectron );
      }
    }


    // ------------------------------------------------------
    // Fill histograms if the candidate is a tau
    // ------------------------------------------------------
    if      ( pctau != 0 ) {
//        cout << "Filling tau" << endl;
       // Here is where we do not yet have a histogram for this role
      if ( histoTau_.map.find( role ) == histoTau_.map.end() ) {
	histoTau_.map[role] = new HistoTau( dir_, role, role ) ;
      }
      // Here is if the candidate is a shallow clone, we need to
      // fill kinematic information from the shallow clone and 
      // detector information from the base object
      if ( c->hasMasterClone() ) {
	histoTau_.map[role]    ->fill( pshallow );
      }
      // Here is if the candidate is a straightforward pointer
      else {
	histoTau_.map[role]    ->fill( pctau );
      }
    }


    // ------------------------------------------------------
    // Fill histograms if the candidate is a jet
    // ------------------------------------------------------
    if      ( pcjet != 0 ) {
//        cout << "Filling jet" << endl;
       // Here is where we do not yet have a histogram for this role
      if ( histoJet_.map.find( role ) == histoJet_.map.end() ) {
	histoJet_.map[role] = new HistoJet( dir_, role, role ) ;
      }
      // Here is if the candidate is a shallow clone, we need to
      // fill kinematic information from the shallow clone and 
      // detector information from the base object
      if ( c->hasMasterClone() ) {
	histoJet_.map[role]    ->fill( pshallow );
      }
      // Here is if the candidate is a straightforward pointer
      else {
	histoJet_.map[role]    ->fill( pcjet );
      }
    }


    // ------------------------------------------------------
    // Fill histograms if the candidate is a met
    // ------------------------------------------------------
    if      ( pcmet != 0 ) {
//        cout << "Filling met" << endl;
       // Here is where we do not yet have a histogram for this role
      if ( histoMET_.map.find( role ) == histoMET_.map.end() ) {
	histoMET_.map[role] = new HistoMET( dir_, role, role ) ;
      }
      // Here is if the candidate is a shallow clone, we need to
      // fill kinematic information from the shallow clone and 
      // detector information from the base object
      if ( c->hasMasterClone() ) {
	histoMET_.map[role]    ->fill( pshallow );
      }
      // Here is if the candidate is a straightforward pointer
      else {
	histoMET_.map[role]    ->fill( pcmet );
      }
    }



    // ------------------------------------------------------
    // Fill histograms if the candidate is a photon
    // ------------------------------------------------------
    if      ( pcphoton != 0 ) {
//        cout << "Filling photon" << endl;
       // Here is where we do not yet have a histogram for this role
      if ( histoPhoton_.map.find( role ) == histoPhoton_.map.end() ) {
	histoPhoton_.map[role] = new HistoPhoton( dir_, role, role ) ;
      }
      // Here is if the candidate is a shallow clone, we need to
      // fill kinematic information from the shallow clone and 
      // detector information from the base object
      if ( c->hasMasterClone() ) {
	histoPhoton_.map[role]    ->fill( pshallow );
      }
      // Here is if the candidate is a straightforward pointer
      else {
	histoPhoton_.map[role]    ->fill( pcphoton );
      }
    }

    // ------------------------------------------------------
    // Fill histograms if the candidate is a track
    // ------------------------------------------------------
    if      ( pctrack != 0 ) {
//        cout << "Filling track" << endl;
       // Here is where we do not yet have a histogram for this role
      if ( histoTrack_.map.find( role ) == histoTrack_.map.end() ) {
	histoTrack_.map[role] = new HistoTrack( dir_, role, role ) ;
      }
      // Here is if the candidate is a shallow clone, we need to
      // fill kinematic information from the shallow clone and 
      // detector information from the base object
      if ( c->hasMasterClone() ) {
	histoTrack_.map[role]    ->fill( pshallow );
      }
      // Here is if the candidate is a straightforward pointer
      else {
	histoTrack_.map[role]    ->fill( pctrack );
      }
    }

    // ------------------------------------------------------
    // Fill histograms if the candidate is a composite
    // ------------------------------------------------------
    if      ( pccomposite != 0 ) {
//        cout << "Filling composite" << endl;
       // Here is where we do not yet have a histogram for this role
      if ( histoComposite_.map.find( role ) == histoComposite_.map.end() ) {
	histoComposite_.map[role] = new HistoComposite( dir_, role, role ) ;
      }
      histoComposite_.map[role]    ->fill( pccomposite );
    }


  }
}
