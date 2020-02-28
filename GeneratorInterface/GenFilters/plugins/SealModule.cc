#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaFilter.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaDauFilter.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaProbeFilter.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaFilterGammaGamma.h"
#include "GeneratorInterface/GenFilters/plugins/JetFlavourFilter.h"
#include "GeneratorInterface/GenFilters/plugins/MCDijetResonance.h"
#include "GeneratorInterface/GenFilters/plugins/MCPdgIndexFilter.h"
#include "GeneratorInterface/GenFilters/plugins/MCSingleParticleFilter.h"
#include "GeneratorInterface/GenFilters/plugins/RecoDiMuon.h"
#include "GeneratorInterface/GenFilters/plugins/MCLongLivedParticles.h"
#include "GeneratorInterface/GenFilters/plugins/MCParticlePairFilter.h"
#include "GeneratorInterface/GenFilters/plugins/CosmicGenFilterHelix.h"
#include "GeneratorInterface/GenFilters/plugins/CosmicGenFilterLowE.h"
#include "GeneratorInterface/GenFilters/plugins/BHFilter.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaFilterIsolatedTrack.h"
#include "GeneratorInterface/GenFilters/plugins/doubleEMEnrichingFilter.h"
#include "GeneratorInterface/GenFilters/plugins/MCDecayingPionKaonFilter.h"
#include "GeneratorInterface/GenFilters/plugins/ComphepSingletopFilter.h"
#include "GeneratorInterface/GenFilters/plugins/STFilter.h"
#include "GeneratorInterface/GenFilters/plugins/LQGenFilter.h"
#include "GeneratorInterface/GenFilters/plugins/XtoFFbarFilter.h"
#include "GeneratorInterface/GenFilters/plugins/TwoVBGenFilter.h"
#include "GeneratorInterface/GenFilters/plugins/TotalKinematicsFilter.h"
#include "GeneratorInterface/GenFilters/plugins/LHEDYdecayFilter.h"
#include "GeneratorInterface/GenFilters/plugins/Zto2lFilter.h"
#include "GeneratorInterface/GenFilters/plugins/HeavyQuarkFromMPIFilter.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaDauVFilterMatchID.h"

using cms::BHFilter;
DEFINE_FWK_MODULE(LQGenFilter);
DEFINE_FWK_MODULE(PythiaFilter);
DEFINE_FWK_MODULE(PythiaDauFilter);
DEFINE_FWK_MODULE(PythiaProbeFilter);
DEFINE_FWK_MODULE(PythiaFilterGammaGamma);
DEFINE_FWK_MODULE(JetFlavourFilter);
DEFINE_FWK_MODULE(MCDijetResonance);
DEFINE_FWK_MODULE(MCPdgIndexFilter);
DEFINE_FWK_MODULE(MCSingleParticleFilter);
DEFINE_FWK_MODULE(RecoDiMuon);
DEFINE_FWK_MODULE(MCLongLivedParticles);
DEFINE_FWK_MODULE(MCParticlePairFilter);
DEFINE_FWK_MODULE(CosmicGenFilterHelix);
DEFINE_FWK_MODULE(CosmicGenFilterLowE);
DEFINE_FWK_MODULE(BHFilter);
DEFINE_FWK_MODULE(PythiaFilterIsolatedTrack);
DEFINE_FWK_MODULE(doubleEMEnrichingFilter);
DEFINE_FWK_MODULE(MCDecayingPionKaonFilter);
DEFINE_FWK_MODULE(ComphepSingletopFilter);
DEFINE_FWK_MODULE(STFilter);
DEFINE_FWK_MODULE(XtoFFbarFilter);
DEFINE_FWK_MODULE(TwoVBGenFilter);
DEFINE_FWK_MODULE(TotalKinematicsFilter);
DEFINE_FWK_MODULE(LHEDYdecayFilter);
DEFINE_FWK_MODULE(Zto2lFilter);
DEFINE_FWK_MODULE(HeavyQuarkFromMPIFilter);
DEFINE_FWK_MODULE(PythiaDauVFilterMatchID);
