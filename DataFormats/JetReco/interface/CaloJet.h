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
 * \version   $Id: CaloJet.h,v 1.17 2006/10/20 08:18:28 llista Exp $
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/JetReco/interface/CaloJetfwd.h"

namespace reco {
class CaloJet : public Jet {
 public:
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
	 mN90 (0)
    {}
    
    /// Maximum energy in EM towers
    double mMaxEInEmTowers;
    /// Maximum energy in HCAL towers
    double mMaxEInHadTowers;
    /// Hadronic nergy fraction in HO
    double mHadEnergyInHO;
    /// Hadronic energy in HB
    double mHadEnergyInHB;
    /// Hadronic energy in HF
    double mHadEnergyInHF;
    /// Hadronic energy in HE
    double mHadEnergyInHE;
    /// Em energy in EB
    double mEmEnergyInEB;
    /// Em energy in EE
    double mEmEnergyInEE;
    /// Em energy in HF
    double mEmEnergyInHF;
    /// Hadronic energy fraction
    double mEnergyFractionHadronic;
    /// Em energy fraction
    double mEnergyFractionEm;
    /// Number of constituents carrying 90% of the Jet energy
    int mN90;
  };
  
  /** Default constructor*/
  CaloJet() {}
  
  /** Constructor from values*/
  CaloJet(const LorentzVector& fP4, const Point& fVertex, const Specific& fSpecific, 
	  const std::vector<CaloTowerDetId>& fIndices);

  /** backward compatible, vertex=(0,0,0) */
  CaloJet(const LorentzVector& fP4, const Specific& fSpecific, 
	  const std::vector<CaloTowerDetId>& fIndices);

  
  virtual ~CaloJet() {};
  
  /** Returns the maximum energy deposited in ECAL towers*/
  double maxEInEmTowers() const {return m_specific.mMaxEInEmTowers;};
  /** Returns the maximum energy deposited in HCAL towers*/
  double maxEInHadTowers() const {return m_specific.mMaxEInHadTowers;};
  /** Returns the jet hadronic energy fraction*/
  double energyFractionHadronic () const {return m_specific.mEnergyFractionHadronic;};
  /** Returns the jet electromagnetic energy fraction*/
  double emEnergyFraction() const {return m_specific.mEnergyFractionEm;};
  /** Returns the jet hadronic energy in HB*/ 
  double hadEnergyInHB() const {return m_specific.mHadEnergyInHB;};
  /** Returns the jet hadronic energy in HO*/
  double hadEnergyInHO() const {return m_specific.mHadEnergyInHO;};
  /** Returns the jet hadronic energy in HE*/
  double hadEnergyInHE() const {return m_specific.mHadEnergyInHE;};
  /** Returns the jet hadronic energy in HF*/
  double hadEnergyInHF() const {return m_specific.mHadEnergyInHF;};
  /** Returns the jet electromagnetic energy in EB*/
  double emEnergyInEB() const {return m_specific.mEmEnergyInEB;};
  /** Returns the jet electromagnetic energy in EE*/
  double emEnergyInEE() const {return m_specific.mEmEnergyInEE;};
  /** Returns the jet electromagnetic energy extracted from HF*/
  double emEnergyInHF() const {return m_specific.mEmEnergyInHF;};
  /** Returns the number of constituents carrying a 90% of the total Jet energy*/
  int n90() const {return m_specific.mN90;};

  // constituents
  CaloTowerRef getConstituent (unsigned fIndex) const;
  std::vector <CaloTowerRef> getConstituents () const;
  
  // block accessors
  
  const std::vector<CaloTowerDetId>& getTowerIndices() const {return m_towerIdxs;};
  const Specific& getSpecific () const {return m_specific;}

  /// Polymorphic clone
  virtual CaloJet* clone () const;

  /// Print object
  virtual std::string print () const;
  
 private:
  /// Polymorphic overlap
  virtual bool overlap( const Candidate & ) const;
  
  // Data members
  /** List of CaloTower IDs the Jet consists of*/
  std::vector<CaloTowerDetId> m_towerIdxs;
  //Variables specific to to the CaloJet class
  Specific m_specific;
};
}
#endif
