#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJet.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterZJet.h"
#include "GeneratorInterface/GenFilters/interface/PythiaHLTSoupFilter.h"
#include "GeneratorInterface/GenFilters/interface/BsJpsiPhiFilter.h"
#include "GeneratorInterface/GenFilters/interface/JetFlavourFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJetWithBg.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJetWithOutBg.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterZJetWithOutBg.h"

 

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilter);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterGammaJet);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterZJet);
  DEFINE_ANOTHER_FWK_MODULE(PythiaHLTSoupFilter);
  DEFINE_ANOTHER_FWK_MODULE(BsJpsiPhiFilter);
  DEFINE_ANOTHER_FWK_MODULE(JetFlavourFilter);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterGammaJetWithBg);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterGammaJetWithOutBg);
  DEFINE_ANOTHER_FWK_MODULE(PythiaFilterZJetWithOutBg);

