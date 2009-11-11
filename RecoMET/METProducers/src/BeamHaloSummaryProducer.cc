#include "RecoMET/METProducers/interface/BeamHaloSummaryProducer.h"

/*
  [class]:  BeamHaloSummaryProducer
  [authors]: R. Remington, The University of Florida
  [description]: See BeamHaloSummaryProducer.h
  [date]: October 15, 2009
*/

using namespace edm;
using namespace std;
using namespace reco;

BeamHaloSummaryProducer::BeamHaloSummaryProducer(const edm::ParameterSet& iConfig)
{
  IT_CSCHaloData = iConfig.getParameter<edm::InputTag>("CSCHaloDataLabel");
  IT_EcalHaloData = iConfig.getParameter<edm::InputTag>("EcalHaloDataLabel");
  IT_HcalHaloData = iConfig.getParameter<edm::InputTag>("HcalHaloDataLabel");
  IT_GlobalHaloData = iConfig.getParameter<edm::InputTag>("GlobalHaloDataLabel");

  produces<BeamHaloSummary>();
}

void BeamHaloSummaryProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  // CSC Specific Halo Data
  Handle<CSCHaloData> TheCSCData;
  iEvent.getByLabel(IT_CSCHaloData, TheCSCData);
  
  // Ecal Specific Halo Data
  Handle<EcalHaloData> TheEcalData;
  iEvent.getByLabel(IT_EcalHaloData, TheEcalData);
  
  // Hcal Specific Halo Data
  Handle<HcalHaloData> TheHcalData;
  iEvent.getByLabel(IT_HcalHaloData, TheHcalData) ;
  
  // Global Halo Data
  Handle<GlobalHaloData> TheGlobalData;
  iEvent.getByLabel(IT_GlobalHaloData, TheGlobalData);

  // Store it to the BeamHaloSummary object and put it in the event
  std::auto_ptr<BeamHaloSummary> TheBeamHaloSummary( new BeamHaloSummary() );
  iEvent.put(TheBeamHaloSummary);
  return;
}

void BeamHaloSummaryProducer::beginJob(){return;}
void BeamHaloSummaryProducer::endJob(){return;}
void BeamHaloSummaryProducer::beginRun(edm::Run&, const edm::EventSetup&){return;}
void BeamHaloSummaryProducer::endRun(edm::Run&, const edm::EventSetup&){return;}
BeamHaloSummaryProducer::~BeamHaloSummaryProducer(){}
