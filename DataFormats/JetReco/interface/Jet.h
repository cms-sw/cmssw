#ifndef JetReco_Jet_h
#define JetReco_Jet_h

 /** \class reco::Jet
 *
 * \short Base class for all types of Jets
 *
 * Jet describes properties common for all kinds of jets, 
 * essentially kinematics. Base class for all types of Jets.
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   Original: April 22, 2005 by Fernando Varela Rodriguez.
 * \version   May 23, 2006 by F.R.
 * \version   $Id: Jet.h,v 1.16 2007/05/30 22:06:42 fedor Exp $
 ************************************************************/
#include <string>
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h"

namespace reco {
  class Jet : public CompositeRefCandidate {
  public:
    typedef std::vector<reco::CandidateRef> Constituents;

    /// record to store eta-phi first and second moments
    class EtaPhiMoments {
    public:
      float etaMean;
      float phiMean;
      float etaEtaMoment;
      float phiPhiMoment;
      float etaPhiMoment;
    };

    /// Default constructor
    Jet () {}
    /// Initiator
    Jet (const LorentzVector& fP4, const Point& fVertex, const Constituents& fConstituents);
    /// Destructor
    virtual ~Jet () {}

    /// eta-phi statistics, ET weighted
    EtaPhiMoments etaPhiStatistics () const;

    /// eta-eta second moment, ET weighted
    float etaetaMoment () const;

    /// phi-phi second moment, ET weighted
    float phiphiMoment () const;

    /// eta-phi second moment, ET weighted
    float etaphiMoment () const;

    /// ET in annulus between rmin and rmax around jet direction
    float etInAnnulus (float fRmin, float fRmax) const;

    /// return # of constituent carrying fraction of energy
    int nCarrying (float fFraction) const;
 
    /// maximum distance from jet to constituent
    float maxDistance () const;
 
    /// # of constituents
    virtual int nConstituents () const {return numberOfDaughters();}

    /// Physics Eta (use jet Z and kinematics only)
    virtual float physicsEtaQuick (float fZVertex) const;

    /// Physics Eta (loop over constituents)
    virtual float physicsEtaDetailed (float fZVertex) const;

    /// list of constituents
    Constituents getJetConstituents () const;

    // quick list of constituents
    std::vector<const reco::Candidate*> getJetConstituentsQuick () const;

  /// Print object
    virtual std::string print () const;

  /// temporary fix for cached valuse
    double massUncached() const {return p4().M();}
    double massSqrUncached() const {return p4().M2();}
    double mtUncached() const {return p4().Mt();}
    double mtSqrUncached() const {return p4().Mt2();}
    double ptUncached() const {return p4().Pt();}
    double phiUncached() const {return p4().Phi();}
    double etaUncached() const {return p4().Eta();}
    double rapidityUncached() const {return p4().Rapidity();}
    double yUncached() const {return p4().Rapidity();}

  private:
    // disallow constituents modifications
    void addDaughter( const CandidateRef & fRef) {CompositeRefCandidate::addDaughter (fRef);}
  };
}
#endif
