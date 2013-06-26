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
// $Id: SiStripElectron.h,v 1.15 2012/10/14 08:45:37 innocent Exp $
//

// system include files

#include <vector>

// user include files

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RefVector.h" 

// forward declarations

namespace reco {
  
  class SiStripElectron : public RecoCandidate {
  public:
    /// default constructor
    SiStripElectron() : RecoCandidate() { }
    /// constructor from band algorithm
    SiStripElectron(const reco::SuperClusterRef& superCluster,
		    Charge q,
		    const std::vector<SiStripRecHit2D>& rphiRecHits,
		    const std::vector<SiStripRecHit2D>& stereoRecHits,
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
      : RecoCandidate(q, PtEtaPhiMass(pt,etaFromRZ(pt,pz), phiAtOrigin, 0.000510f), Point(0,0,0), -11 * q )
	, superCluster_(superCluster)
      , rphiRecHits_(rphiRecHits)
      , stereoRecHits_(stereoRecHits)
      , superClusterPhiVsRSlope_(superClusterPhiVsRSlope)
      , phiVsRSlope_(phiVsRSlope)
      , phiAtOrigin_(phiAtOrigin)
      , chi2_(chi2)
      , ndof_(ndof)
      , zVsRSlope_(zVsRSlope)
      , numberOfStereoHits_(numberOfStereoHits)
      , numberOfBarrelRphiHits_(numberOfBarrelRphiHits)
      , numberOfEndcapZphiHits_(numberOfEndcapZphiHits) { }
        
    /// constructor from RecoCandidate
    template<typename P4>
    SiStripElectron( Charge q, const P4 & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx, -11 * q ) { }
    /// destructor
    virtual ~SiStripElectron();
    /// returns a clone of the candidate
    virtual SiStripElectron * clone() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    
    /// reference to the rphiRecHits identified as belonging to an electron
    const std::vector<SiStripRecHit2D>& rphiRecHits() const { return rphiRecHits_; }
    /// reference to the stereoRecHits identified as belonging to an electron
    const std::vector<SiStripRecHit2D>& stereoRecHits() const { return stereoRecHits_; }
    
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
        
    /// returns z(r) slope fit from stereo tracker hits (constrained to pass through supercluster)
    double zVsRSlope() const { return zVsRSlope_; }
    
    /// returns number of stereo hits in phi band (barrel + endcap)
    unsigned int numberOfStereoHits() const { return numberOfStereoHits_; }
    /// returns number of barrel rphi hits in phi band
    unsigned int numberOfBarrelRphiHits() const { return numberOfBarrelRphiHits_; }
    /// returns number of endcap zphi hits in phi band
    unsigned int numberOfEndcapZphiHits() const { return numberOfEndcapZphiHits_; }

    bool isElectron() const;
  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    std::vector<SiStripRecHit2D> rphiRecHits_;
    std::vector<SiStripRecHit2D> stereoRecHits_;
    
    double superClusterPhiVsRSlope_;
    double phiVsRSlope_;
    double phiAtOrigin_;
    double chi2_;
    int ndof_;
    
    double zVsRSlope_;
    
    unsigned int numberOfStereoHits_;
    unsigned int numberOfBarrelRphiHits_;
    unsigned int numberOfEndcapZphiHits_;
   };
}

#endif
