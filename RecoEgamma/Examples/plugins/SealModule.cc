//#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

#include "ElectronIDAnalyzer.h"
#include "ElectronSeedAnalyzer.h"
#include "MCElectronAnalyzer.h"
#include "MCPhotonAnalyzer.h"
#include "MCPizeroAnalyzer.h"
#include "SimpleConvertedPhotonAnalyzer.h"
#include "SimplePhotonAnalyzer.h"
#include "SiStripElectronAnalyzer.h"
#include "GsfElectronMCAnalyzer.h"
#include "GsfElectronDataAnalyzer.h"
#include "GsfElectronFakeAnalyzer.h"
#include "GsfElectronMCFakeAnalyzer.h"
#include "PatPhotonSimpleAnalyzer.h"


#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

typedef Merger<reco::SuperClusterCollection> EgammaSuperClusterMerger;
DEFINE_FWK_MODULE(EgammaSuperClusterMerger);
DEFINE_FWK_MODULE(ElectronIDAnalyzer);
DEFINE_FWK_MODULE(ElectronSeedAnalyzer);
DEFINE_FWK_MODULE(MCElectronAnalyzer);
DEFINE_FWK_MODULE(MCPhotonAnalyzer);
DEFINE_FWK_MODULE(MCPizeroAnalyzer);
DEFINE_FWK_MODULE(GsfElectronMCAnalyzer);
DEFINE_FWK_MODULE(GsfElectronDataAnalyzer);
DEFINE_FWK_MODULE(GsfElectronFakeAnalyzer);
DEFINE_FWK_MODULE(GsfElectronMCFakeAnalyzer);
DEFINE_FWK_MODULE(SimpleConvertedPhotonAnalyzer);
DEFINE_FWK_MODULE(SimplePhotonAnalyzer);
DEFINE_FWK_MODULE(SiStripElectronAnalyzer);
DEFINE_FWK_MODULE(PatPhotonSimpleAnalyzer);
