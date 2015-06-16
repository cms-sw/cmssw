#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Modules/interface/JetCorrectionESProducer.h"
#include "JetMETCorrections/Modules/interface/JetCorrectionESSource.h"
#include "JetMETCorrections/Modules/interface/JetCorrectionESChain.h"
#include "JetMETCorrections/Modules/interface/JetCorrectionProducer.h"
#include "JetMETCorrections/Modules/interface/QGLikelihoodESProducer.h"
#include "JetMETCorrections/Modules/interface/QGLikelihoodSystematicsESProducer.h"
#include "JetMETCorrections/Algorithms/interface/LXXXCorrector.h"
#include "JetMETCorrections/Algorithms/interface/L1OffsetCorrector.h"
#include "JetMETCorrections/Algorithms/interface/L1JPTOffsetCorrector.h"
#include "JetMETCorrections/Algorithms/interface/L1FastjetCorrector.h"
#include "JetMETCorrections/Algorithms/interface/L6SLBCorrector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "JetMETCorrections/Objects/interface/METCorrectionsRecord.h"
#include "CondFormats/JetMETObjects/interface/METCorrectorParameters.h"


REGISTER_PLUGIN(JetCorrectionsRecord,JetCorrectorParametersCollection);
REGISTER_PLUGIN(METCorrectionsRecord,METCorrectorParametersCollection);

using namespace cms;
using namespace reco;

typedef JetCorrectionProducer<CaloJet> CaloJetCorrectionProducer;
DEFINE_FWK_MODULE(CaloJetCorrectionProducer);

typedef JetCorrectionProducer<PFJet> PFJetCorrectionProducer;
DEFINE_FWK_MODULE(PFJetCorrectionProducer);

typedef JetCorrectionProducer<JPTJet> JPTJetCorrectionProducer;
DEFINE_FWK_MODULE(JPTJetCorrectionProducer);

typedef JetCorrectionProducer<TrackJet> TrackJetCorrectionProducer;
DEFINE_FWK_MODULE(TrackJetCorrectionProducer);

typedef JetCorrectionProducer<GenJet> GenJetCorrectionProducer;
DEFINE_FWK_MODULE(GenJetCorrectionProducer);

typedef JetCorrectionProducer<BasicJet> BasicJetCorrectionProducer;
DEFINE_FWK_MODULE(BasicJetCorrectionProducer);

DEFINE_FWK_EVENTSETUP_MODULE(JetCorrectionESChain);


DEFINE_FWK_EVENTSETUP_MODULE(QGLikelihoodESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(QGLikelihoodSystematicsESProducer);


//--------------- Generic LX corrections --------------------
DEFINE_JET_CORRECTION_ESSOURCE (LXXXCorrector, LXXXCorrectionESSource);
DEFINE_JET_CORRECTION_ESPRODUCER (LXXXCorrector, LXXXCorrectionESProducer);

//--------------- L1 Offset subtraction corrections ---------
DEFINE_JET_CORRECTION_ESSOURCE (L1OffsetCorrector, L1OffsetCorrectionESSource);
DEFINE_JET_CORRECTION_ESPRODUCER (L1OffsetCorrector, L1OffsetCorrectionESProducer);

//--------------- L1 Offset subtraction corrections ---------
DEFINE_JET_CORRECTION_ESSOURCE (L1JPTOffsetCorrector, L1JPTOffsetCorrectionESSource);
DEFINE_JET_CORRECTION_ESPRODUCER (L1JPTOffsetCorrector, L1JPTOffsetCorrectionESProducer);

//--------------- L1 fastjet UE&PU subtraction corrections --
DEFINE_JET_CORRECTION_ESSOURCE (L1FastjetCorrector, L1FastjetCorrectionESSource);
DEFINE_JET_CORRECTION_ESPRODUCER (L1FastjetCorrector, L1FastjetCorrectionESProducer);

//---------------  L6 SLB corrections -----------------------
DEFINE_JET_CORRECTION_ESSOURCE (L6SLBCorrector, L6SLBCorrectionESSource);
DEFINE_JET_CORRECTION_ESPRODUCER (L6SLBCorrector, L6SLBCorrectionESProducer);
