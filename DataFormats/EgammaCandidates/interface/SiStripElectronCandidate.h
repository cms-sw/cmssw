#ifndef EgammaCandidates_SiStripElectronCandidate_h
#define EgammaCandidates_SiStripElectronCandidate_h
// -*- C++ -*-
//
// Package:     EgammaCandidates
// Class  :     SiStripElectronCandidate
// 
/**\class SiStripElectronCandidate SiStripElectronCandidate.h DataFormats/EgammaCandidates/interface/SiStripElectronCandidate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 15:43:14 EDT 2006
// $Id$
//

// system include files

// user include files

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronCandidateFwd.h"

// forward declarations

namespace reco {
   class SiStripElectronCandidate : public RecoCandidate {
      public:
	 /// default constructor
	 SiStripElectronCandidate() : RecoCandidate() { }
	 /// constructor from values
	 SiStripElectronCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
	    RecoCandidate( q, p4, vtx ) { }
	 /// destructor
	 virtual ~SiStripElectronCandidate();
	 /// returns a clone of the candidate
	 virtual SiStripElectronCandidate * clone() const;
	 /// refrence to a Track
	 virtual reco::TrackRef track() const;
	 /// reference to a SuperCluster
	 virtual reco::SuperClusterRef superCluster() const;
	 /// set refrence to Photon component
	 void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
	 /// set refrence to Track component
	 void setTrack( const reco::TrackRef & r ) { track_ = r; }
	 
      private:
	 /// check overlap with another candidate
	 virtual bool overlap( const Candidate & ) const;
	 /// reference to a SuperCluster
	 reco::SuperClusterRef superCluster_;
	 /// reference to a Track
	 reco::TrackRef track_;
   };
}

#endif
