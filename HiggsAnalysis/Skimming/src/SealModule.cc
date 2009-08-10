#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsHLTAnalysis.h>   
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkimProducer.h>
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkimFilter.h>
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkim.h>
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkimEff.h>
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsPreFilter.h>
#include <HiggsAnalysis/Skimming/interface/HiggsToWW2LeptonsSkim.h>
#include <HiggsAnalysis/Skimming/interface/HeavyChHiggsToTauNuSkim.h>
#include <HiggsAnalysis/Skimming/interface/HiggsTo2GammaSkim.h>
#include <HiggsAnalysis/Skimming/interface/HiggsToTauTauElectronTauSkim.h>
#include <HiggsAnalysis/Skimming/interface/HiggsToTauTauMuonTauSkim.h>
#include <HiggsAnalysis/Skimming/interface/LightChHiggsToTauNuSkim.h>


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HiggsToZZ4LeptonsHLTAnalysis);
DEFINE_ANOTHER_FWK_MODULE(HiggsToZZ4LeptonsSkimProducer);
DEFINE_ANOTHER_FWK_MODULE(HiggsToZZ4LeptonsSkimFilter);
DEFINE_ANOTHER_FWK_MODULE(HiggsToZZ4LeptonsSkim);
DEFINE_ANOTHER_FWK_MODULE(HiggsToZZ4LeptonsSkimEff);
DEFINE_ANOTHER_FWK_MODULE(HiggsToZZ4LeptonsPreFilter);
DEFINE_ANOTHER_FWK_MODULE(HiggsToWW2LeptonsSkim);
DEFINE_ANOTHER_FWK_MODULE(HeavyChHiggsToTauNuSkim);
DEFINE_ANOTHER_FWK_MODULE(HiggsTo2GammaSkim);
DEFINE_ANOTHER_FWK_MODULE(HiggsToTauTauElectronTauSkim);
DEFINE_ANOTHER_FWK_MODULE(HiggsToTauTauMuonTauSkim);
DEFINE_ANOTHER_FWK_MODULE(LightChHiggsToTauNuSkim);

