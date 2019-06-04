#ifndef METReco_GenMET_h
#define METReco_GenMET_h

/** \class GenMET
 *
 * \short MET made from Generator level HEPMC particles
 *
 * GenMET represents MET made from HEPMC particles
 * Provide energy contributions from different particles
 * in addition to generic MET parameters
 *
 * \author    R. Cavanaugh, UFL (inspiration taken from F. Ratnikov)
 *
 ************************************************************/

#include "DataFormats/METReco/interface/SpecificGenMETData.h"
#include "DataFormats/METReco/interface/MET.h"

namespace reco {
  class GenMET : public MET {
  public:
    /* Constructors*/
    GenMET();
    GenMET(const SpecificGenMETData& gen_data_, double sumet_, const LorentzVector& fP4, const Point& fVertex)
        : MET(sumet_, fP4, fVertex), gen_data(gen_data_) {}
    /* Default destructor*/
    ~GenMET() override {}

    //Get Neutral EM Et Fraction
    double NeutralEMEtFraction() const { return gen_data.NeutralEMEtFraction; }

    //Get Neutral EM Et
    double NeutralEMEt() const { return gen_data.NeutralEMEtFraction * sumEt(); }

    //Get Charged EM Et Fraction
    double ChargedEMEtFraction() const { return gen_data.ChargedEMEtFraction; }

    //Get Charged EM Et
    double ChargedEMEt() const { return gen_data.ChargedEMEtFraction * sumEt(); }

    //Get Neutral Had Et Fraction
    double NeutralHadEtFraction() const { return gen_data.NeutralHadEtFraction; }

    //Get Neutral Had Et
    double NeutralHadEt() const { return gen_data.NeutralHadEtFraction * sumEt(); }

    //Get Charged Had Et Fraction
    double ChargedHadEtFraction() const { return gen_data.ChargedHadEtFraction; }

    //Get Charged Had Et
    double ChargedHadEt() const { return gen_data.ChargedHadEtFraction * sumEt(); }

    //Get Muon Et Fraction
    double MuonEtFraction() const { return gen_data.MuonEtFraction; }

    //Get Muon Et
    double MuonEt() const { return gen_data.MuonEtFraction * sumEt(); }

    //Get Invisible Et Fraction
    double InvisibleEtFraction() const { return gen_data.InvisibleEtFraction; }

    //Get Invisible Et
    double InvisibleEt() const { return gen_data.InvisibleEtFraction * sumEt(); }

    // Old Accessors (to be removed as soon as possible)
    /** Returns energy of electromagnetic particles*/
    double emEnergy() const { return gen_data.m_EmEnergy; };
    /** Returns energy of hadronic particles*/
    double hadEnergy() const { return gen_data.m_HadEnergy; };
    /** Returns invisible energy*/
    double invisibleEnergy() const { return gen_data.m_InvisibleEnergy; };
    /** Returns other energy (undecayed Sigmas etc.)*/
    double auxiliaryEnergy() const { return gen_data.m_AuxiliaryEnergy; };
    // block accessors

    // block accessors
  private:
    bool overlap(const Candidate&) const override;
    // Data members
    //Variables specific to to the GenMET class
    SpecificGenMETData gen_data;
  };
}  // namespace reco
#endif
