#ifndef EgammaCandidates_SiStripElectron_h
#define EgammaCandidates_SiStripElectron_h
// -*- C++ -*-
//
// Package:     EgammaCandidates
// Class  :     SiStripElectron
// 
/**\class SiStripElectron SiStripElectron.h DataFormats/EgammaCandidates/interface/SiStripElectron.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 15:43:14 EDT 2006
// $Id: SiStripElectron.h,v 1.2 2006/06/02 22:43:01 pivarski Exp $
//

// system include files

#include <vector>

// user include files

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"

// forward declarations

namespace reco {
   class SiStripElectron : public RecoCandidate {
      public:
	 /// default constructor
	 SiStripElectron() : RecoCandidate() { }
	 /// constructor from band algorithm
	 SiStripElectron(const reco::SuperClusterRef& superCluster,
			 Charge q,
			 double superClusterPhiVsRSlope,
			 double phiVsRSlope,
			 double phiAtOrigin,
			 double chi2,
			 int ndof,
			 double pt,
			 double pz,
			 double zVsRSlope,
			 unsigned int numberOfStereoHits,
			 unsigned int numberOfBarrelRphiHits,
			 unsigned int numberOfEndcapZphiHits)
	    : RecoCandidate(q, LorentzVector(pt*cos(phiAtOrigin), pt*sin(phiAtOrigin), pz, sqrt(pt*pt+pz*pz+0.000510*0.000510)), Point(0,0,0))
	    , superCluster_(superCluster)
	    , superClusterPhiVsRSlope_(superClusterPhiVsRSlope)
	    , phiVsRSlope_(phiVsRSlope)
	    , phiAtOrigin_(phiAtOrigin)
	    , chi2_(chi2)
	    , ndof_(ndof)
	    , pt_(pt)
	    , pz_(pz)
	    , zVsRSlope_(zVsRSlope)
	    , numberOfStereoHits_(numberOfStereoHits)
	    , numberOfBarrelRphiHits_(numberOfBarrelRphiHits)
	    , numberOfEndcapZphiHits_(numberOfEndcapZphiHits) { }
	 /// copy constructor (update in SiStripElectron.cc)
	 SiStripElectron(const SiStripElectron& rhs)
	    : RecoCandidate(rhs.charge(), rhs.p4(), rhs.vertex())
	    , superCluster_(rhs.superCluster())
	    , superClusterPhiVsRSlope_(rhs.superClusterPhiVsRSlope())
	    , phiVsRSlope_(rhs.phiVsRSlope())
	    , phiAtOrigin_(rhs.phiAtOrigin())
	    , chi2_(rhs.chi2())
	    , ndof_(rhs.ndof())
	    , pt_(rhs.pt())
	    , pz_(rhs.pz())
	    , zVsRSlope_(rhs.zVsRSlope())
	    , numberOfStereoHits_(rhs.numberOfStereoHits())
	    , numberOfBarrelRphiHits_(rhs.numberOfBarrelRphiHits())
	    , numberOfEndcapZphiHits_(rhs.numberOfEndcapZphiHits()) { }
	 /// constructor from RecoCandidate
	 SiStripElectron( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
	    RecoCandidate( q, p4, vtx ) { }
	 /// destructor
	 virtual ~SiStripElectron();
	 /// returns a clone of the candidate
	 virtual SiStripElectron * clone() const;
	 /// reference to a SuperCluster
	 virtual reco::SuperClusterRef superCluster() const;
	 /// set refrence to SuperCluster
	 void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }

	 /// returns phi(r) projection from supercluster
	 double superClusterPhiVsRSlope() const { return superClusterPhiVsRSlope_; }
	 /// returns phi(r) slope from fit to tracker hits
	 double phiVsRSlope() const { return phiVsRSlope_; }
	 /// returns phi(r=0) intercept from fit to tracker hits
	 double phiAtOrigin() const { return phiAtOrigin_; }
	 /// returns chi^2 of fit to tracker hits
	 double chi2() const { return chi2_; }
	 /// returns number of degrees of freedom of fit to tracker hits
	 int ndof() const { return ndof_; }

	 /// returns transverse momentum, as determined by fit to tracker hits
	 double pt() const { return pt_; }
	 /// returns longitudinal momentum, as determined by fit to tracker hits
	 double pz() const { return pz_; }

	 /// returns z(r) slope fit from stereo tracker hits (constrained to pass through supercluster)
	 double zVsRSlope() const { return zVsRSlope_; }

	 /// returns number of stereo hits in phi band (barrel + endcap)
	 unsigned int numberOfStereoHits() const { return numberOfStereoHits_; }
	 /// returns number of barrel rphi hits in phi band
	 unsigned int numberOfBarrelRphiHits() const { return numberOfBarrelRphiHits_; }
	 /// returns number of endcap zphi hits in phi band
	 unsigned int numberOfEndcapZphiHits() const { return numberOfEndcapZphiHits_; }

      private:
	 /// check overlap with another candidate
	 virtual bool overlap( const Candidate & ) const;
	 /// reference to a SuperCluster
	 reco::SuperClusterRef superCluster_;

	 double superClusterPhiVsRSlope_;
	 double phiVsRSlope_;
	 double phiAtOrigin_;
	 double chi2_;
	 int ndof_;

	 double pt_;
	 double pz_;

	 double zVsRSlope_;

	 unsigned int numberOfStereoHits_;
	 unsigned int numberOfBarrelRphiHits_;
	 unsigned int numberOfEndcapZphiHits_;
   };
}

#endif
