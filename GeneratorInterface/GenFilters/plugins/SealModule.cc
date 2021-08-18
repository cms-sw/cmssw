#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaFilter.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaDauFilter.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaProbeFilter.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaFilterGammaGamma.h"
#include "GeneratorInterface/GenFilters/plugins/MCPdgIndexFilter.h"
#include "GeneratorInterface/GenFilters/plugins/MCSingleParticleFilter.h"
#include "GeneratorInterface/GenFilters/plugins/MCParticlePairFilter.h"
#include "GeneratorInterface/GenFilters/plugins/CosmicGenFilterHelix.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaFilterIsolatedTrack.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaDauVFilterMatchID.h"
#include "GeneratorInterface/GenFilters/plugins/PythiaFilterMotherSister.h"

DEFINE_FWK_MODULE(PythiaFilter);
DEFINE_FWK_MODULE(PythiaDauFilter);
DEFINE_FWK_MODULE(PythiaProbeFilter);
DEFINE_FWK_MODULE(PythiaFilterGammaGamma);
DEFINE_FWK_MODULE(MCPdgIndexFilter);
DEFINE_FWK_MODULE(MCSingleParticleFilter);
DEFINE_FWK_MODULE(MCParticlePairFilter);
DEFINE_FWK_MODULE(CosmicGenFilterHelix);
DEFINE_FWK_MODULE(PythiaFilterIsolatedTrack);
DEFINE_FWK_MODULE(PythiaDauVFilterMatchID);
DEFINE_FWK_MODULE(PythiaFilterMotherSister);
