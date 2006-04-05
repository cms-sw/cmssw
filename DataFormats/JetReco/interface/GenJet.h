#ifndef JetReco_GenJet_h
#define JetReco_GenJet_h

/** \class GenJet
 *
 * \short Jets made from MC generator particles
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   1st Version March 31, 2006
 ************************************************************/


#include "DataFormats/JetReco/interface/CommonJetData.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include <vector>
#include "DataFormats/JetReco/interface/GenJetfwd.h"

typedef std::vector<GenJet> GenJetCollection;

class GenJet : public Jet {
public:
  struct Specific {
    Specific () :
      m_EmEnergy (0),
	 m_HadEnergy (0),
	 m_InvisibleEnergy (0) {}

    /// Energy of EM particles
    double m_EmEnergy;
    /// Energy of Hadrons
    double m_HadEnergy;
    /// Invisible energy (mu, nu, ...)
    double m_InvisibleEnergy;
  };

  /** Default constructor*/
  GenJet() {}

  /** Constructor from values*/
  GenJet(const CommonJetData& fCommon, const Specific& fSpecific, 
	  const std::vector<int>& fBarcodes):
    m_data (fCommon), m_barcodes (fBarcodes), m_specific (fSpecific)  {}
  
  /** Default destructor*/
  virtual ~GenJet() {};

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


  //These methods are specific to the GenJet class
  
  /** Returns energy of electromagnetic particles*/
  double emEnergy() const {return m_specific.m_EmEnergy;};
  /** Returns energy of hadronic particles*/
  double hadEnergy() const {return m_specific.m_HadEnergy;};
  /** Returns invisible energy*/
  double invisibleEnergy() const {return m_specific.m_InvisibleEnergy;};

  // block accessors

  const std::vector<int>& getBarcodes() const {return m_barcodes;};
  const CommonJetData& getCommonData () const {return m_data;}
  const Specific& getSpecific () const {return m_specific;}

  
private:
  // Data members
  /** Structure containing data common to all types of jets*/
  CommonJetData m_data;
  /** List of MC particles the Jet consists of*/
  std::vector<int> m_barcodes;
  //Variables specific to to the GenJet class
  Specific m_specific;
};
#endif
