#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/ImpactParameter/interface/JetProbabilityComputer.h"
#include "RecoBTag/ImpactParameter/interface/JetBProbabilityComputer.h"
#include "RecoBTag/ImpactParameter/interface/CandidateJetProbabilityComputer.h"
#include "RecoBTag/ImpactParameter/interface/CandidateJetBProbabilityComputer.h"
#include "RecoBTag/ImpactParameter/plugins/IPProducer.h"
#include "RecoBTag/ImpactParameter/interface/NegativeTrackCountingComputer.h"
#include "RecoBTag/ImpactParameter/interface/TrackCountingComputer.h"
#include "RecoBTag/ImpactParameter/interface/CandidateNegativeTrackCountingComputer.h"
#include "RecoBTag/ImpactParameter/interface/CandidateTrackCountingComputer.h"
#include "RecoBTag/ImpactParameter/interface/PromptTrackCountingComputer.h"


//DEFINE_FWK_MODULE(TrackIPProducer);

typedef IPProducer<reco::TrackRefVector,reco::JTATagInfo, IPProducerHelpers::FromJTA> TrackIPProducer;
DEFINE_FWK_MODULE(TrackIPProducer);
typedef IPProducer<std::vector<reco::CandidatePtr>,reco::JetTagInfo,  IPProducerHelpers::FromJetAndCands> CandIPProducer;
DEFINE_FWK_MODULE(CandIPProducer);

typedef JetTagComputerESProducer<TrackCountingComputer>       TrackCountingESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(TrackCountingESProducer);
typedef JetTagComputerESProducer<NegativeTrackCountingComputer>       NegativeTrackCountingESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(NegativeTrackCountingESProducer);

typedef JetTagComputerESProducer<CandidateTrackCountingComputer>          CandidateTrackCountingESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateTrackCountingESProducer);
typedef JetTagComputerESProducer<CandidateNegativeTrackCountingComputer>  CandidateNegativeTrackCountingESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateNegativeTrackCountingESProducer);

typedef JetTagComputerESProducer<JetProbabilityComputer>       JetProbabilityESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(JetProbabilityESProducer);
typedef JetTagComputerESProducer<JetBProbabilityComputer>       JetBProbabilityESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(JetBProbabilityESProducer);

typedef JetTagComputerESProducer<CandidateJetProbabilityComputer>       CandidateJetProbabilityESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateJetProbabilityESProducer);
typedef JetTagComputerESProducer<CandidateJetBProbabilityComputer>       CandidateJetBProbabilityESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateJetBProbabilityESProducer);

typedef JetTagComputerESProducer<PromptTrackCountingComputer>  PromptTrackCountingESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(PromptTrackCountingESProducer);

