#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJet.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterZJet.h"
#include "GeneratorInterface/GenFilters/interface/PythiaHLTSoupFilter.h"
#include "GeneratorInterface/GenFilters/interface/BsJpsiPhiFilter.h"
#include "GeneratorInterface/GenFilters/interface/JetFlavourFilter.h"
#include "GeneratorInterface/GenFilters/interface/JetFlavourCutFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJetWithBg.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJetWithOutBg.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterZJetWithOutBg.h"
#include "GeneratorInterface/GenFilters/interface/MCDijetResonance.h"
#include "GeneratorInterface/GenFilters/interface/MCProcessFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCProcessRangeFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCSingleParticleFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCZll.h"
#include "GeneratorInterface/GenFilters/interface/MinimumBiasFilter.h"
#include "GeneratorInterface/GenFilters/interface/RecoDiMuon.h"
#include "GeneratorInterface/GenFilters/interface/MCParticlePairFilter.h"
#include "GeneratorInterface/GenFilters/interface/CosmicGenFilterHelix.h"
#include "GeneratorInterface/GenFilters/interface/CosmicGenFilterLowE.h"
<<<<<<< SealModule.cc
#include "GeneratorInterface/GenFilters/interface/BHFilter.h"
=======
#include "GeneratorInterface/GenFilters/interface/PythiaFilterEMJet.h"
#include "GeneratorInterface/GenFilters/interface/HZZ4lFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaGamma.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterZgamma.h"
#include "GeneratorInterface/GenFilters/interface/BdecayFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJetIsoPi0.h"
#include "GeneratorInterface/GenFilters/interface/Zto2lFilter.h"
>>>>>>> 1.15

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilter);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterGammaJet);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterZJet);
  DEFINE_ANOTHER_FWK_MODULE(PythiaHLTSoupFilter);
  DEFINE_ANOTHER_FWK_MODULE(BsJpsiPhiFilter);
  DEFINE_ANOTHER_FWK_MODULE(JetFlavourFilter);
  DEFINE_ANOTHER_FWK_MODULE(JetFlavourCutFilter);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterGammaJetWithBg);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterGammaJetWithOutBg);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterZJetWithOutBg);
  DEFINE_ANOTHER_FWK_MODULE(MCDijetResonance);
  DEFINE_ANOTHER_FWK_MODULE(MCProcessFilter);
  DEFINE_ANOTHER_FWK_MODULE(MCProcessRangeFilter);
  DEFINE_ANOTHER_FWK_MODULE(MCSingleParticleFilter);
  DEFINE_ANOTHER_FWK_MODULE(MCZll);
  DEFINE_ANOTHER_FWK_MODULE(MinimumBiasFilter);
  DEFINE_ANOTHER_FWK_MODULE(RecoDiMuon);
  DEFINE_ANOTHER_FWK_MODULE(MCParticlePairFilter);
  DEFINE_ANOTHER_FWK_MODULE(CosmicGenFilterHelix);
  DEFINE_ANOTHER_FWK_MODULE(CosmicGenFilterLowE);
<<<<<<< SealModule.cc
  DEFINE_ANOTHER_FWK_MODULE(BHFilter);


=======
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterEMJet);
  DEFINE_ANOTHER_FWK_MODULE(HZZ4lFilter);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterGammaGamma);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterZgamma);
  DEFINE_ANOTHER_FWK_MODULE(BdecayFilter);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterGammaJetIsoPi0);
  DEFINE_ANOTHER_FWK_MODULE(Zto2lFilter);
>>>>>>> 1.15
