#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/TrackProbability/interface/JetProbabilityComputer.h"
#include "RecoBTag/TrackProbability/interface/JetBProbabilityComputer.h"
//#include "RecoBTag/TrackProbability/interface/JetMassProbabilityComputer.h"
#include "RecoBTag/ImpactParameter/plugins/TrackIPProducer.h"



DEFINE_FWK_MODULE(TrackIPProducer);


typedef JetTagComputerESProducer<JetProbabilityComputer>       JetProbabilityESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(JetProbabilityESProducer);
typedef JetTagComputerESProducer<JetBProbabilityComputer>       JetBProbabilityESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(JetBProbabilityESProducer);


