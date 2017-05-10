#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterMultiMother.h"
#include "GeneratorInterface/GenFilters/interface/PythiaDauFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJet.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaGamma.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterZJet.h"
#include "GeneratorInterface/GenFilters/interface/PythiaHLTSoupFilter.h"
#include "GeneratorInterface/GenFilters/interface/BsJpsiPhiFilter.h"
#include "GeneratorInterface/GenFilters/interface/JetFlavourFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJetWithBg.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJetWithOutBg.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterZJetWithOutBg.h"
#include "GeneratorInterface/GenFilters/interface/MCDijetResonance.h"
#include "GeneratorInterface/GenFilters/interface/MCProcessFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCProcessRangeFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCPdgIndexFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCSingleParticleFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCSmartSingleParticleFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCZll.h"
#include "GeneratorInterface/GenFilters/interface/MinimumBiasFilter.h"
#include "GeneratorInterface/GenFilters/interface/RecoDiMuon.h"
#include "GeneratorInterface/GenFilters/interface/MCLongLivedParticles.h"
#include "GeneratorInterface/GenFilters/interface/MCParticlePairFilter.h"
#include "GeneratorInterface/GenFilters/interface/CosmicGenFilterHelix.h"
#include "GeneratorInterface/GenFilters/interface/CosmicGenFilterLowE.h"
#include "GeneratorInterface/GenFilters/interface/BHFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterIsolatedTrack.h"
#include "GeneratorInterface/GenFilters/interface/BCToEFilter.h"
#include "GeneratorInterface/GenFilters/interface/EMEnrichingFilter.h"
#include "GeneratorInterface/GenFilters/interface/doubleEMEnrichingFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCDecayingPionKaonFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterEMJetHeep.h"
#include "GeneratorInterface/GenFilters/interface/ComphepSingletopFilter.h"
#include "GeneratorInterface/GenFilters/interface/STFilter.h"
#include "GeneratorInterface/GenFilters/interface/PythiaFilterTTBar.h"
#include "GeneratorInterface/GenFilters/interface/LQGenFilter.h"
#include "GeneratorInterface/GenFilters/interface/XtoFFbarFilter.h"
#include "GeneratorInterface/GenFilters/interface/HerwigMaxPtPartonFilter.h"
#include "GeneratorInterface/GenFilters/interface/TwoVBGenFilter.h"
#include "GeneratorInterface/GenFilters/interface/TotalKinematicsFilter.h"
#include "GeneratorInterface/GenFilters/interface/LHEDYdecayFilter.h"
#include "GeneratorInterface/GenFilters/interface/Zto2lFilter.h"
#include "GeneratorInterface/GenFilters/interface/ZgMassFilter.h"
#include "GeneratorInterface/GenFilters/interface/ZgammaMassFilter.h"
#include "GeneratorInterface/GenFilters/interface/HeavyQuarkFromMPIFilter.h"

  using cms::BHFilter;
  DEFINE_FWK_MODULE(LQGenFilter);
  DEFINE_FWK_MODULE(PythiaFilter);
  DEFINE_FWK_MODULE(PythiaFilterMultiMother);
  DEFINE_FWK_MODULE(PythiaDauFilter);
  DEFINE_FWK_MODULE(PythiaFilterGammaJet);
  DEFINE_FWK_MODULE(PythiaFilterGammaGamma);
  DEFINE_FWK_MODULE(PythiaFilterZJet);
  DEFINE_FWK_MODULE(PythiaHLTSoupFilter);
  DEFINE_FWK_MODULE(BsJpsiPhiFilter);
  DEFINE_FWK_MODULE(JetFlavourFilter);
  DEFINE_FWK_MODULE(PythiaFilterGammaJetWithBg);
  DEFINE_FWK_MODULE(PythiaFilterGammaJetWithOutBg);
  DEFINE_FWK_MODULE(PythiaFilterZJetWithOutBg);
  DEFINE_FWK_MODULE(MCDijetResonance);
  DEFINE_FWK_MODULE(MCProcessFilter);
  DEFINE_FWK_MODULE(MCProcessRangeFilter);
  DEFINE_FWK_MODULE(MCPdgIndexFilter);
  DEFINE_FWK_MODULE(MCSingleParticleFilter);
  DEFINE_FWK_MODULE(MCSmartSingleParticleFilter);
  DEFINE_FWK_MODULE(MCZll);
  DEFINE_FWK_MODULE(MinimumBiasFilter);
  DEFINE_FWK_MODULE(RecoDiMuon);
  DEFINE_FWK_MODULE(MCLongLivedParticles);
  DEFINE_FWK_MODULE(MCParticlePairFilter);
  DEFINE_FWK_MODULE(CosmicGenFilterHelix);
  DEFINE_FWK_MODULE(CosmicGenFilterLowE);
  DEFINE_FWK_MODULE(BHFilter);
  DEFINE_FWK_MODULE(PythiaFilterIsolatedTrack);
  DEFINE_FWK_MODULE(BCToEFilter);
  DEFINE_FWK_MODULE(EMEnrichingFilter);
  DEFINE_FWK_MODULE(doubleEMEnrichingFilter);
  DEFINE_FWK_MODULE(MCDecayingPionKaonFilter);
  DEFINE_FWK_MODULE(PythiaFilterEMJetHeep);
  DEFINE_FWK_MODULE(ComphepSingletopFilter);
  DEFINE_FWK_MODULE(STFilter);
  DEFINE_FWK_MODULE(PythiaFilterTTBar);
  DEFINE_FWK_MODULE(XtoFFbarFilter);
  DEFINE_FWK_MODULE(HerwigMaxPtPartonFilter);
  DEFINE_FWK_MODULE(TwoVBGenFilter);
  DEFINE_FWK_MODULE(TotalKinematicsFilter);
  DEFINE_FWK_MODULE(LHEDYdecayFilter);
  DEFINE_FWK_MODULE(Zto2lFilter);
  DEFINE_FWK_MODULE(ZgMassFilter);
  DEFINE_FWK_MODULE(ZgammaMassFilter);
  DEFINE_FWK_MODULE(HeavyQuarkFromMPIFilter);
