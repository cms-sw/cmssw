#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/ImpactParameter/interface/JetProbabilityComputer.h"
#include "RecoBTag/ImpactParameter/interface/JetBProbabilityComputer.h"
//#include "RecoBTag/TrackProbability/interface/JetMassProbabilityComputer.h"
#include "RecoBTag/ImpactParameter/plugins/TrackIPProducer.h"
#include "RecoBTag/ImpactParameter/interface/NegativeTrackCountingComputer.h"
#include "RecoBTag/ImpactParameter/interface/TrackCountingComputer.h"
#include "RecoBTag/ImpactParameter/interface/PromptTrackCountingComputer.h"


DEFINE_FWK_MODULE(TrackIPProducer);

typedef JetTagComputerESProducer<TrackCountingComputer>       TrackCountingESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(TrackCountingESProducer);
typedef JetTagComputerESProducer<NegativeTrackCountingComputer>       NegativeTrackCountingESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(NegativeTrackCountingESProducer);





typedef JetTagComputerESProducer<JetProbabilityComputer>       JetProbabilityESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(JetProbabilityESProducer);
typedef JetTagComputerESProducer<JetBProbabilityComputer>       JetBProbabilityESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(JetBProbabilityESProducer);


typedef JetTagComputerESProducer<PromptTrackCountingComputer>  PromptTrackCountingESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(PromptTrackCountingESProducer);
