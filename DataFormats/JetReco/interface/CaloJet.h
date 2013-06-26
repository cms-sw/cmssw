#ifndef JetReco_CaloJet_h
#define JetReco_CaloJet_h

/** \class reco::CaloJet
 *
 * \short Jets made from CaloTowers
 *
 * CaloJet represents Jets made from CaloTowers
 * Provide energy contributions from different subdetectors
 * in addition to generic Jet parameters
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   Original: April 22, 2005 by Fernando Varela Rodriguez.
 * 
 * \version   Oct 19, 2005, R. Harris, modified to work 
 *            with real CaloTowers. No energy fractions yet.
 *
 * \version   May 3, 2006, F.Ratnikov, include all different
 *            energy components separately
 * \version   $Id: CaloJet.h,v 1.39 2013/05/01 13:53:47 srappocc Exp $
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"


namespace reco {
class CaloJet : public Jet {
 public:

    typedef CaloTowerPtr ConstituentTypePtr;
    typedef CaloTowerFwdPtr ConstituentTypeFwdPtr;


  struct Specific {
    Specific () :
      mMaxEInEmTowers (0),
	 mMaxEInHadTowers (0),
	 mHadEnergyInHO (0),
	 mHadEnergyInHB (0),
	 mHadEnergyInHF (0),
	 mHadEnergyInHE (0),
	 mEmEnergyInEB (0),
	 mEmEnergyInEE (0),
	 mEmEnergyInHF (0),
	 mEnergyFractionHadronic(0),
	 mEnergyFractionEm (0),
	 mTowersArea (0)
    {}
    
    /// Maximum energy in EM towers
    float mMaxEInEmTowers;
    /// Maximum energy in HCAL towers
    float mMaxEInHadTowers;
    /// Hadronic nergy fraction in HO
    float mHadEnergyInHO;
    /// Hadronic energy in HB
    float mHadEnergyInHB;
    /// Hadronic energy in HF
    float mHadEnergyInHF;
    /// Hadronic energy in HE
    float mHadEnergyInHE;
    /// Em energy in EB
    float mEmEnergyInEB;
    /// Em energy in EE
    float mEmEnergyInEE;
    /// Em energy in HF
    float mEmEnergyInHF;
    /// Hadronic energy fraction
    float mEnergyFractionHadronic;
    /// Em energy fraction
    float mEnergyFractionEm;
    /// Area of contributing CaloTowers
    float mTowersArea;
  };
  
  /** Default constructor*/
  CaloJet() {}
  
  /** Constructor from values*/
  CaloJet(const LorentzVector& fP4, const Point& fVertex, const Specific& fSpecific, 
	  const Jet::Constituents& fConstituents);

  /** Constructor from values*/
  CaloJet(const LorentzVector& fP4, const Point& fVertex, const Specific& fSpecific);

  /** backward compatible, vertex=(0,0,0) */
  CaloJet(const LorentzVector& fP4, const Specific& fSpecific, 
	  const Jet::Constituents& fConstituents);

  
  virtual ~CaloJet() {};
  
  /** Returns the maximum energy deposited in ECAL towers*/
  float maxEInEmTowers() const {return m_specific.mMaxEInEmTowers;}
  /** Returns the maximum energy deposited in HCAL towers*/
  float maxEInHadTowers() const {return m_specific.mMaxEInHadTowers;}
  /** Returns the jet hadronic energy fraction*/
  float energyFractionHadronic () const {return m_specific.mEnergyFractionHadronic;}
  /** Returns the jet electromagnetic energy fraction*/
  float emEnergyFraction() const {return m_specific.mEnergyFractionEm;}
  /** Returns the jet hadronic energy in HB*/ 
  float hadEnergyInHB() const {return m_specific.mHadEnergyInHB;}
  /** Returns the jet hadronic energy in HO*/
  float hadEnergyInHO() const {return m_specific.mHadEnergyInHO;}
  /** Returns the jet hadronic energy in HE*/
  float hadEnergyInHE() const {return m_specific.mHadEnergyInHE;}
  /** Returns the jet hadronic energy in HF*/
  float hadEnergyInHF() const {return m_specific.mHadEnergyInHF;}
  /** Returns the jet electromagnetic energy in EB*/
  float emEnergyInEB() const {return m_specific.mEmEnergyInEB;}
  /** Returns the jet electromagnetic energy in EE*/
  float emEnergyInEE() const {return m_specific.mEmEnergyInEE;}
  /** Returns the jet electromagnetic energy extracted from HF*/
  float emEnergyInHF() const {return m_specific.mEmEnergyInHF;}
  /** Returns area of contributing towers */
  float towersArea() const {return m_specific.mTowersArea;}
  /** Returns the number of constituents carrying a 90% of the total Jet energy*/
  int n90() const {return nCarrying (0.9);}
  /** Returns the number of constituents carrying a 60% of the total Jet energy*/
  int n60() const {return nCarrying (0.6);}

  /// Physics Eta (use jet Z and kinematics only)
    //  float physicsEtaQuick (float fZVertex) const;
  /// Physics Eta (use jet Z and kinematics only)
    //float physicsEta (float fZVertex) const {return physicsEtaQuick (fZVertex);}
  /// Physics p4 (use jet Z and kinematics only)
    //LorentzVector physicsP4 (float fZVertex) const;
  /// Physics p4 for full 3d vertex corretion
  LorentzVector physicsP4 (const Particle::Point &vertex) const;
  /// detector p4 for full 3d vertex correction.
  LorentzVector detectorP4 () const;
  
  /// Physics Eta (loop over constituents)
    //float physicsEtaDetailed (float fZVertex) const;

  /// Detector Eta (default for CaloJets)
    //float detectorEta () const {return eta();}


  /// get specific constituent
  virtual CaloTowerPtr getCaloConstituent (unsigned fIndex) const;
  /// get all constituents
  virtual std::vector <CaloTowerPtr> getCaloConstituents () const;
  
  // block accessors
  
  const Specific& getSpecific () const {return m_specific;}

  /// Polymorphic clone
  virtual CaloJet* clone () const;

  /// Print object
  virtual std::string print () const;

  /// CaloTowers indexes
  std::vector<CaloTowerDetId> getTowerIndices() const;
 private:
  /// Polymorphic overlap
  virtual bool overlap( const Candidate & ) const;
  
  //Variables specific to to the CaloJet class
  Specific m_specific;
};
}
// temporary fix before include_checcker runs globally
#include "DataFormats/JetReco/interface/CaloJetCollection.h" //INCLUDECHECKER:SKIP
#endif
