#ifndef JetReco_CaloJet_h
#define JetReco_CaloJet_h

/** \class CaloJet
 *
 * \short Jets made from CaloTowers
 *
 * CaloJet represents Jets made from CaloTowers
 * More to be added...
 *
 * \author Fernando Varela Rodriguez, Boston University
 *
 * \version   1st Version April 22, 2005.
 * 
 * \version   2nd Version Oct 19, 2005, R. Harris, modified to work 
 *            with real CaloTowers. No energy fractions yet.
 *
 * \version   3rd Version May 3, 2006, F.Ratnikov, include all different
 *            energy components separately
 ************************************************************/


#include "DataFormats/JetReco/interface/CommonJetData.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include <vector>

#include "DataFormats/JetReco/interface/CaloJetfwd.h"

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
  CaloJet(const CommonJetData& fCommon, const Specific& fSpecific, 
	  const std::vector<CaloTowerDetId>& fIndices):
    m_data (fCommon), m_towerIdxs (fIndices), m_specific (fSpecific)  {}
  
  /** Default destructor*/
  virtual ~CaloJet() {};

  /// four-momentum Lorentz vector
  virtual LorentzVector p4() const;
  /// spatial momentum vector
  virtual Vector momentum() const;
  /** Rereturns the jet momentum component along the x axis */
  virtual double px() const;
  /** Returns the jet momentum component along the y axis */
  virtual double py() const;
  /** Returns the jet momentum component along the z axis */
  virtual double pz() const;
  /** Returns the total energy of the jet*/
  virtual double energy () const;

  /** Standard quantities derived from the Jet Lorentz vector
  /\return the modulus of the momentum of the jet */
  virtual double p() const;
  /** Returns the transverse momentum of the jet*/
  virtual double pt() const;
  /** Returns the transverse energy of the jet*/
  virtual double et() const;
  /** Returns the jet mass of the jet*/
  virtual double mass() const;
  /** Returns the azimuthal angle of the jet, Phi*/
  virtual double phi() const;
  /** Returns the pseudorapidity of the jet*/
  virtual double eta() const;
  /** Returns the number of constituents of the jet*/
  virtual int nConstituents() const;


  //These methods are specific to the CaloJet class
  
  /** Returns the list of CaloTower IDs forming the Jet*/
  
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

  // block accessors

  const std::vector<CaloTowerDetId>& getTowerIndices() const {return m_towerIdxs;};
  const CommonJetData& getCommonData () const {return m_data;}
  const Specific& getSpecific () const {return m_specific;}

  
private:
  // Data members
  /** Structure containing data common to all types of jets*/
  CommonJetData m_data;
  /** List of CaloTower IDs the Jet consists of*/
  std::vector<CaloTowerDetId> m_towerIdxs;
  //Variables specific to to the CaloJet class
  Specific m_specific;
};
#endif
