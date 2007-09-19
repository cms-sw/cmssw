#include "RecoTracker/MeasurementDet/plugins/MeasurementTrackerSiStripRefGetterProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

//measurement tracker
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/OnDemandMeasurementTracker.h"


using namespace std;
//using namespace sistrip;

// -----------------------------------------------------------------------------
//
MeasurementTrackerSiStripRefGetterProducer::MeasurementTrackerSiStripRefGetterProducer( const edm::ParameterSet& conf ) :
  inputModuleLabel_(conf.getParameter<edm::InputTag>("InputModuleLabel")),
  cabling_(),
  measurementTrackerName_(conf.getParameter<string>("measurementTrackerName"))
{
  produces< RefGetter >();
}

// -----------------------------------------------------------------------------
MeasurementTrackerSiStripRefGetterProducer::~MeasurementTrackerSiStripRefGetterProducer() {}

// -----------------------------------------------------------------------------
void MeasurementTrackerSiStripRefGetterProducer::beginJob( const edm::EventSetup& setup) {
 //get cabling
  setup.get<SiStripRegionCablingRcd>().get(cabling_);
}

// -----------------------------------------------------------------------------
void MeasurementTrackerSiStripRefGetterProducer::endJob() {;}

// -----------------------------------------------------------------------------
/** */
void MeasurementTrackerSiStripRefGetterProducer::produce( edm::Event& event, 
					    const edm::EventSetup& setup ) {

  // Retrieve unpacking tool from event
  edm::Handle< LazyGetter > lazygetter;
  event.getByLabel(inputModuleLabel_,lazygetter);
  
  // Construct default RefGetter object
  std::auto_ptr<RefGetter> refgetter(new RefGetter());

  //retreive the measurement tracker.
  edm::ESHandle<MeasurementTracker> mtESH;
  setup.get<CkfComponentsRecord>().get(measurementTrackerName_,mtESH);
  
  //cast it to the proper type
  const OnDemandMeasurementTracker * tOD = dynamic_cast<const OnDemandMeasurementTracker *>(mtESH.product());
  
  if (!tOD){
    edm::LogError("MeasurementTrackerSiStripRefGetterProducer")<<"casting of MeasurementTracker named: "<<measurementTrackerName_<<" into OnDemandMeasurementTracker does not work.";
  }
  else{
    //define the regions for each individual module
    tOD->define(event, lazygetter, refgetter);
  }

  // Add to event
  event.put(refgetter);
}

