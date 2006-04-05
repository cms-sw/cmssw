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
      m_maxEInEmTowers (0),
	 m_maxEInHadTowers (0),
	 m_energyFractionInHO (0),
	 m_energyFractionInHB (0),
	 m_energyFractionInHF (0),
	 m_energyFractionInHE (0),
	 m_energyFractionInHCAL (0),
	 m_energyFractionInECAL (0) {}

    /// Maximum energy in EM towers
    double m_maxEInEmTowers;
    /// Maximum energy in HCAL towers
    double m_maxEInHadTowers;
    /// Energy fraction in HO
    double m_energyFractionInHO;
    /// Energy fraction in HB
    double m_energyFractionInHB;
    /// Energy fraction in HF
    double m_energyFractionInHF;
    /// Energy fraction in HE
    double m_energyFractionInHE;
    /// Energy fraction in HCAL
    double m_energyFractionInHCAL;
    /// Energy fraction in ECAL
    double m_energyFractionInECAL;
    /// Number of constituents carrying 90% of the Jet energy
    int m_n90;
  };

  /** Default constructor*/
  CaloJet() {}

  /** Constructor from values*/
  CaloJet(const CommonJetData& fCommon, const Specific& fSpecific, 
	  const std::vector<CaloTowerDetId>& fIndices):
    m_data (fCommon), m_towerIdxs (fIndices), m_specific (fSpecific)  {}
  
  /** Default destructor*/
  virtual ~CaloJet() {};

  /** The Jet four-vector as a true Lorentz vector
  \return the jet momentum component along the x axis */
  virtual double getPx() const;
  /** Returns the jet momentum component along the y axis */
  virtual double getPy() const;
  /** Returns the jet momentum component along the z axis */
  virtual double getPz() const;
  /** Returns the total energy of the jet*/
  virtual double getE() const;

  /** Standard quantities derived from the Jet Lorentz vector
  /\return the modulus of the momentum of the jet */
  virtual double getP() const;
  /** Returns the transverse momentum of the jet*/
  virtual double getPt() const;
  /** Returns the transverse energy of the jet*/
  virtual double getEt() const;
  /** Returns the jet mass of the jet*/
  virtual double getM() const;
  /** Returns the azimuthal angle of the jet, Phi*/
  virtual double getPhi() const;
  /** Returns the pseudorapidity of the jet*/
  virtual double getEta() const;
  /** Returns the rapidity of the jet*/
  virtual double getY() const;
  /** Returns the number of constituents of the jet*/
  virtual int getNConstituents() const;


  //These methods are specific to the CaloJet class
  
  /** Returns the list of CaloTower IDs forming the Jet*/
  
  /** Returns the maximum energy deposited in ECAL towers*/
  double maxEInEmTowers() const {return m_specific.m_maxEInEmTowers;};
  /** Returns the maximum energy deposited in HCAL towers*/
  double maxEInHadTowers() const {return m_specific.m_maxEInHadTowers;};
  /** Returns the jet energy fraction in HCAL*/
  double energyFractionInHCAL() const {return m_specific.m_energyFractionInHCAL;};
  /** Returns the jet energy fraction in ECAL*/
  double energyFractionInECAL() const {return m_specific.m_energyFractionInECAL;};
  /** Returns the jet energy fraction in HB*/ 
  double energyFractionInHB() const {return m_specific.m_energyFractionInHB;};
  /** Returns the jet energy fraction in HO*/
  double energyFractionInHO() const {return m_specific.m_energyFractionInHO;};
  /** Returns the jet energy fraction in HES*/
  double energyFractionInHE() const {return m_specific.m_energyFractionInHE;};
   /** Returns the jet energy fraction in HF*/
  double energyFractionInHF() const {return m_specific.m_energyFractionInHF;};
  /** Returns the number of constituents carrying a 90% of the total Jet energy*/
  int n90() const {return m_specific.m_n90;};

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
