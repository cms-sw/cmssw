#ifndef JetReco_GenJet_h
#define JetReco_GenJet_h

/** \class GenJet
 *
 * \short Jets made from MC generator particles
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   1st Version March 31, 2006
 * $Id$
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/JetReco/interface/GenJetfwd.h"

class GenJet : public Jet {
public:
  struct Specific {
    Specific () :
      m_EmEnergy (0),
	 m_HadEnergy (0),
	 m_InvisibleEnergy (0),
	 m_AuxiliaryEnergy (0) {}

    /// Energy of EM particles
    double m_EmEnergy;
    /// Energy of Hadrons
    double m_HadEnergy;
    /// Invisible energy (mu, nu, ...)
    double m_InvisibleEnergy;
    /// Anything else (undecayed Sigmas etc.)
    double m_AuxiliaryEnergy;
  };

  /** Default constructor*/
  GenJet() {}

  /** Constructor from values*/
  GenJet(const LorentzVector& fP4, const Specific& fSpecific, 
	 const std::vector<int>& fBarcodes);

  virtual ~GenJet() {};
  /** Returns energy of electromagnetic particles*/
  double emEnergy() const {return m_specific.m_EmEnergy;};
  /** Returns energy of hadronic particles*/
  double hadEnergy() const {return m_specific.m_HadEnergy;};
  /** Returns invisible energy*/
  double invisibleEnergy() const {return m_specific.m_InvisibleEnergy;};
  /** Returns other energy (undecayed Sigmas etc.*/
  double auxiliaryEnergy() const {return m_specific.m_AuxiliaryEnergy;};

  // block accessors

  const std::vector<int>& getBarcodes() const {return m_barcodes;};
  const Specific& getSpecific () const {return m_specific;}

  
private:
  // Data members
  /** List of MC particles the Jet consists of*/
  std::vector<int> m_barcodes;
  //Variables specific to to the GenJet class
  Specific m_specific;
};
#endif
