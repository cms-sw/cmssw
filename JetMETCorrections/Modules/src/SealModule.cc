#include "CondCore/PluginSystem/interface/registration_macros.h"
DEFINE_SEAL_MODULE();

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
REGISTER_PLUGIN (JetCorrectionsRecord, JetCorrector);

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

using namespace cms;
using namespace reco;

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "JetMETCorrections/Modules/src/JetCorrectionProducer.h"
typedef JetCorrectionProducer<CaloJet> CaloJetCorrectionProducer;
DEFINE_ANOTHER_FWK_MODULE(CaloJetCorrectionProducer);
typedef JetCorrectionProducer<PFJet> PFJetCorrectionProducer;
DEFINE_ANOTHER_FWK_MODULE(PFJetCorrectionProducer);
#include "PlotJetCorrections.h"
DEFINE_ANOTHER_FWK_MODULE(PlotJetCorrections);
#include "JetCorrectionServiceChain.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(JetCorrectionServiceChain);

#include "JetCorrectionService.icc"
#include "JetMETCorrections/Objects/interface/SimpleJetCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (SimpleJetCorrector, SimpleJetCorrectionService);
#include "JetMETCorrections/Algorithms/interface/MCJetCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (MCJetCorrector, MCJetCorrectionService);
#include "JetMETCorrections/Algorithms/interface/MCJetCorrector3D.h"
DEFINE_JET_CORRECTION_SERVICE (MCJetCorrector3D, MCJetCorrectionService3D);
#include "JetMETCorrections/Algorithms/interface/ZSPJetCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (ZSPJetCorrector, ZSPJetCorrectionService);
#include "JetMETCorrections/Algorithms/interface/L2RelativeCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (L2RelativeCorrector, L2RelativeCorrectionService);
#include "JetMETCorrections/Algorithms/interface/L3AbsoluteCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (L3AbsoluteCorrector, L3AbsoluteCorrectionService);
#include "JetMETCorrections/Algorithms/interface/L3PFAbsoluteCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (L3PFAbsoluteCorrector, L3PFAbsoluteCorrectionService);
#include "JetMETCorrections/Algorithms/interface/L4EMFCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (L4EMFCorrector, L4EMFCorrectionService);
#include "JetMETCorrections/Algorithms/interface/L5FlavorCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (L5FlavorCorrector, L5FlavorCorrectionService);
#include "JetMETCorrections/Algorithms/interface/L7PartonCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (L7PartonCorrector, L7PartonCorrectionService);
#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (JetPlusTrackCorrector, JetPlusTrackCorrectionService);

