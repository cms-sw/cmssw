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
 * \version   $Id: Jet.h,v 1.25 2008/04/18 21:58:34 fedor Exp $
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
    Jet (const LorentzVector& fP4, const Point& fVertex);
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

    /// static function to convert detector eta to physics eta
    static float physicsEta (float fZVertex, float fDetectorEta);

    /// static function to convert physics eta to detector eta
    static float detectorEta (float fZVertex, float fPhysicsEta);

    /// list of constituents
    virtual Constituents getJetConstituents () const;

    /// quick list of constituents
    virtual std::vector<const reco::Candidate*> getJetConstituentsQuick () const;

    /// Print object
    virtual std::string print () const;

    /// scale energy of the jet
    virtual void scaleEnergy (double fScale);
    
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

    bool isJet() const;
    
  private:
    float mJetArea;
    float mPileupEnergy;
    int mPassNumber;
  };
}
#endif
