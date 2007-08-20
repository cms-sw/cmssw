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
 * \version   $Id: Jet.h,v 1.18 2007/08/15 17:43:13 fedor Exp $
 ************************************************************/
#include <string>
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"

namespace reco {
  class Jet : public CompositeRefBaseCandidate {
  public:
    typedef reco::CandidateBaseRefVector Constituents;

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
    Jet () : mJetArea (0), mPileupEnergy (0), mPassNumber (0) {}
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

    /// quick list of constituents
    std::vector<const reco::Candidate*> getJetConstituentsQuick () const;

    /// Print object
    virtual std::string print () const;
    
    /// set jet area
    virtual void setJetArea (float fArea) {mJetArea = fArea;}
    /// get jet area
    virtual float jetArea () const {return mJetArea;}

    ///  Set pileup energy contribution as calculated by algorithm
    virtual void setPileup (float fEnergy) {mPileupEnergy = fEnergy;}
    ///  pileup energy contribution as calculated by algorithm
    virtual float pileup () const {return mPileupEnergy;}
    
    ///  Set number of passes taken by algorithm
    virtual void setNPasses (int fPasses) {mPassNumber = fPasses;}
    ///  number of passes taken by algorithm
    virtual int nPasses () const {return mPassNumber;}
    
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
    void addDaughter( const CandidateBaseRef & fRef) {CompositeRefBaseCandidate::addDaughter (fRef);}
    float mJetArea;
    float mPileupEnergy;
    int mPassNumber;
  };
}
#endif
