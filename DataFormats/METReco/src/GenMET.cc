#include "DataFormats/METReco/interface/GenMET.h"

using namespace reco;

GenMET::GenMET() {
  gen_data.NeutralEMEtFraction = 0.0;
  gen_data.NeutralHadEtFraction = 0.0;
  gen_data.ChargedEMEtFraction = 0.0;
  gen_data.ChargedHadEtFraction = 0.0;
  gen_data.MuonEtFraction = 0.0;
  gen_data.InvisibleEtFraction = 0.0;

  //obsolete variables that need to be removed in 4_X_Y
  gen_data.m_EmEnergy = 0.0;
  gen_data.m_HadEnergy = 0.0;
  gen_data.m_InvisibleEnergy = 0.0;
  gen_data.m_AuxiliaryEnergy = 0.0;
}

bool GenMET::overlap(const Candidate&) const { return false; }
