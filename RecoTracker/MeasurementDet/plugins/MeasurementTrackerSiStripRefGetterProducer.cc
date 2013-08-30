#include "RecoTracker/MeasurementDet/plugins/MeasurementTrackerSiStripRefGetterProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

//measurement tracker
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "OnDemandMeasurementTracker.h"


using namespace std;
//using namespace sistrip;

// -----------------------------------------------------------------------------
//
MeasurementTrackerSiStripRefGetterProducer::MeasurementTrackerSiStripRefGetterProducer( const edm::ParameterSet& conf ) :
  MeasurementTrackerEventProducer(conf),
  inputModuleLabel_(conf.getParameter<edm::InputTag>("InputModuleLabel")),
  cabling_()
{
  produces< RefGetter >();
  // produces< MeasurementTrackerEvent >(); // already done by base class
}

// -----------------------------------------------------------------------------
MeasurementTrackerSiStripRefGetterProducer::~MeasurementTrackerSiStripRefGetterProducer() {}

// -----------------------------------------------------------------------------
void MeasurementTrackerSiStripRefGetterProducer::beginRun( edm::Run const&, const edm::EventSetup& setup) {
 //get cabling
  setup.get<SiStripRegionCablingRcd>().get(cabling_);
}

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
  setup.get<CkfComponentsRecord>().get(measurementTrackerLabel_,mtESH);
  
  // create new data structures from templates
  std::auto_ptr<StMeasurementDetSet> stripData(new StMeasurementDetSet(mtESH->stripDetConditions()));
  std::auto_ptr<PxMeasurementDetSet> pixelData(new PxMeasurementDetSet(mtESH->pixelDetConditions()));
  
  //cast MT to the proper type
  const OnDemandMeasurementTracker * tOD = dynamic_cast<const OnDemandMeasurementTracker *>(mtESH.product());
  if (!tOD){
    edm::LogError("MeasurementTrackerSiStripRefGetterProducer")<<"casting of MeasurementTracker named: "<< measurementTrackerLabel_ <<" into OnDemandMeasurementTracker does not work.";
  } else{
    //define the regions for each individual module
    tOD->define(lazygetter, *refgetter, *stripData);
  }

  // Add to event
  edm::OrphanHandle<edm::RefGetter<SiStripCluster> > refgetterH = event.put(refgetter);

  //std::cout << "Created new OnDemand strip data @" << &* stripData << std::endl;
  std::vector<bool> stripClustersToSkip;
  std::vector<bool> pixelClustersToSkip;

  // fill them
  stripData->setLazyGetter(lazygetter);
  stripData->setRefGetter(*refgetterH);
  updateStrips(event, *stripData, stripClustersToSkip);
  updatePixels(event, *pixelData, pixelClustersToSkip);

  // put into MTE
  std::auto_ptr<MeasurementTrackerEvent> out(new MeasurementTrackerEvent(*mtESH, stripData.release(), pixelData.release(), stripClustersToSkip, pixelClustersToSkip));

  // put into event
  event.put(out);

}

void 
MeasurementTrackerSiStripRefGetterProducer::updateStrips( const edm::Event& event, StMeasurementDetSet & theStDets, std::vector<bool> & stripClustersToSkip ) const
{
  LogDebug(category_)<<"Updating siStrip on event: "<< (unsigned int) event.id().run() <<" : "<<(unsigned int) event.id().event();

  //get the skip clusters
  if (selfUpdateSkipClusters_){
    edm::Handle< edm::ContainerMask<edm::LazyGetter<SiStripCluster> > > theStripClusterMask;
    event.getByLabel(pset_.getParameter<edm::InputTag>("skipClusters"), theStripClusterMask);
    theStripClusterMask->copyMaskTo(stripClustersToSkip);
  } else {
    stripClustersToSkip.clear();
  }

  //get the detid that are inactive
  theStDets.rawInactiveStripDetIds().clear();
  getInactiveStrips(event,theStDets.rawInactiveStripDetIds());
}


