#ifndef JetReco_GenJet_h
#define JetReco_GenJet_h

/** \class reco::GenJet
 *
 * \short Jets made from MC generator particles
 *
 * GenJet represents Jets made from MC candidates
 * Provide energy contributions from different particle types
 * in addition to generic Jet parameters
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   Original March 31, 2006 by F.R. 
 *            Added GenJet specifics, 2019 by Salvatore Rappoccio
 ************************************************************/

#include "DataFormats/JetReco/interface/Jet.h"

namespace reco {
  class GenParticle;

  class GenJet : public Jet {
  public:
    struct Specific {
      Specific()
          : m_EmEnergy(0),
            m_HadEnergy(0),
            m_InvisibleEnergy(0),
            m_AuxiliaryEnergy(0),
            m_ChargedHadronEnergy(0),
            m_NeutralHadronEnergy(0),
            m_ChargedEmEnergy(0),
            m_NeutralEmEnergy(0),
            m_MuonEnergy(0),
            m_ChargedHadronMultiplicity(0),
            m_NeutralHadronMultiplicity(0),
            m_ChargedEmMultiplicity(0),
            m_NeutralEmMultiplicity(0),
            m_MuonMultiplicity(0) {}

      /// Calo-like definitions:
      /// Energy of EM particles
      float m_EmEnergy;
      /// Energy of Hadrons
      float m_HadEnergy;
      /// Invisible energy (mu, nu, ...)
      float m_InvisibleEnergy;
      /// Anything else (undecayed Sigmas etc.)
      float m_AuxiliaryEnergy;

      /// PF-like definitions:
      /// pi+, K+, etc
      float m_ChargedHadronEnergy;
      /// K0, etc
      float m_NeutralHadronEnergy;
      /// Electrons
      float m_ChargedEmEnergy;
      /// Photons
      float m_NeutralEmEnergy;
      /// Muons
      float m_MuonEnergy;
      /// Corresponding multiplicities:
      int m_ChargedHadronMultiplicity;
      int m_NeutralHadronMultiplicity;
      int m_ChargedEmMultiplicity;
      int m_NeutralEmMultiplicity;
      int m_MuonMultiplicity;
    };

    /** Default constructor*/
    GenJet() {}

    /** Constructor from values*/
    GenJet(const LorentzVector& fP4,
           const Point& fVertex,
           const Specific& fSpecific,
           const Jet::Constituents& fConstituents);
    GenJet(const LorentzVector& fP4, const Point& fVertex, const Specific& fSpecific);

    /** backward compatible, vertex=(0,0,0) */
    GenJet(const LorentzVector& fP4, const Specific& fSpecific, const Jet::Constituents& fConstituents);

    ~GenJet() override{};
    /** Returns energy of electromagnetic particles*/
    float emEnergy() const { return m_specific.m_EmEnergy; };
    /** Returns energy of hadronic particles*/
    float hadEnergy() const { return m_specific.m_HadEnergy; };
    /** Returns invisible energy*/
    float invisibleEnergy() const { return m_specific.m_InvisibleEnergy; };
    /** Returns other energy (undecayed Sigmas etc.)*/
    float auxiliaryEnergy() const { return m_specific.m_AuxiliaryEnergy; };

    // PF-like definitions
    float chargedHadronEnergy() const { return m_specific.m_ChargedHadronEnergy; }
    float neutralHadronEnergy() const { return m_specific.m_NeutralHadronEnergy; }
    float chargedEmEnergy() const { return m_specific.m_ChargedEmEnergy; }
    float neutralEmEnergy() const { return m_specific.m_NeutralEmEnergy; }
    float muonEnergy() const { return m_specific.m_MuonEnergy; }
    int chargedHadronMultiplicity() const { return m_specific.m_ChargedHadronMultiplicity; }
    int neutralHadronMultiplicity() const { return m_specific.m_NeutralHadronMultiplicity; }
    int chargedEmMultiplicity() const { return m_specific.m_ChargedEmMultiplicity; }
    int neutralEmMultiplicity() const { return m_specific.m_NeutralEmMultiplicity; }
    int muonMultiplicity() const { return m_specific.m_MuonMultiplicity; }

    /// Detector Eta (use reference Z and jet kinematics only)
    float detectorEta(float fZVertex) const;

    /// convert generic constituent to specific type
    static const GenParticle* genParticle(const reco::Candidate* fConstituent);
    /// get specific constituent
    virtual const GenParticle* getGenConstituent(unsigned fIndex) const;
    /// get all constituents
    virtual std::vector<const GenParticle*> getGenConstituents() const;

    // block accessors

    const Specific& getSpecific() const { return m_specific; }

    /// set the specific (note: responsibility of keeping it consistent with the jet daughers belongs to the caller)
    void setSpecific(const Specific& spec) { m_specific = spec; }

    /// Polymorphic clone
    GenJet* clone() const override;

    /// Print object
    std::string print() const override;

  private:
    /// Polymorphic overlap
    bool overlap(const Candidate&) const override;

    // Data members
    //Variables specific to to the GenJet class
    Specific m_specific;
  };
}  // namespace reco
// temporary fix before include_checcker runs globally
#include "DataFormats/JetReco/interface/GenJetCollection.h"  //INCLUDECHECKER:SKIP
#endif
