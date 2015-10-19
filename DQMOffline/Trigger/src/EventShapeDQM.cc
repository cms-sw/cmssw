#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DQMOffline/Trigger/interface/EventShapeDQM.h"

EventShapeDQM::EventShapeDQM(const edm::ParameterSet& ps)
{

}

EventShapeDQM::~EventShapeDQM()
{

}


void EventShapeDQM::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{

}

void EventShapeDQM::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{

}

DEFINE_FWK_MODULE(EventShapeDQM);
