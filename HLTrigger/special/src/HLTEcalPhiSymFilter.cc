#include "HLTrigger/special/interface/HLTEcalPhiSymFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


HLTEcalPhiSymFilter::HLTEcalPhiSymFilter(const edm::ParameterSet& config) :
  barrelDigisToken_( consumes<EBDigiCollection>( config.getParameter<edm::InputTag> ("barrelDigiCollection") ) ),
  endcapDigisToken_( consumes<EEDigiCollection>( config.getParameter<edm::InputTag> ("endcapDigiCollection") ) ),
  barrelUncalibHitsToken_( consumes<EcalUncalibratedRecHitCollection>( config.getParameter<edm::InputTag> ("barrelUncalibHitCollection") ) ),
  endcapUncalibHitsToken_( consumes<EcalUncalibratedRecHitCollection>( config.getParameter<edm::InputTag> ("endcapUncalibHitCollection") ) ),
  barrelHitsToken_( consumes<EBRecHitCollection>( config.getParameter<edm::InputTag> ("barrelHitCollection") ) ),
  endcapHitsToken_( consumes<EERecHitCollection>( config.getParameter<edm::InputTag> ("endcapHitCollection") ) ),
  phiSymBarrelDigis_( config.getParameter<std::string> ("phiSymBarrelDigiCollection") ),
  phiSymEndcapDigis_( config.getParameter<std::string> ("phiSymEndcapDigiCollection") ),
  ampCut_barlP_( config.getParameter<std::vector<double> > ("ampCut_barrelP") ),
  ampCut_barlM_( config.getParameter<std::vector<double> > ("ampCut_barrelM") ),
  ampCut_endcP_( config.getParameter<std::vector<double> > ("ampCut_endcapP") ),
  ampCut_endcM_( config.getParameter<std::vector<double> > ("ampCut_endcapM") ),
  statusThreshold_( config.getParameter<uint32_t> ("statusThreshold") ),
  useRecoFlag_( config.getParameter<bool>("useRecoFlag") )
{
  //register your products
  produces<EBDigiCollection>(phiSymBarrelDigis_);
  produces<EEDigiCollection>(phiSymEndcapDigis_);
}


HLTEcalPhiSymFilter::~HLTEcalPhiSymFilter()
{}

void
HLTEcalPhiSymFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("barrelDigiCollection",edm::InputTag("ecalDigis","ebDigis"));
  desc.add<edm::InputTag>("endcapDigiCollection",edm::InputTag("ecalDigis","eeDigis"));
  desc.add<edm::InputTag>("barrelUncalibHitCollection",edm::InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"));
  desc.add<edm::InputTag>("endcapUncalibHitCollection",edm::InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"));
  desc.add<edm::InputTag>("barrelHitCollection",edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>("endcapHitCollection",edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  desc.add<unsigned int>("statusThreshold",3);
  desc.add<bool>("useRecoFlag",false);
  desc.addOptional<double>("ampCut_barrel",8.)->setComment("Deprecated and to be removed");
  desc.addOptional<double>("ampCut_endcap",12.)->setComment("Deprecated and to be removed");
  desc.add<std::vector<double> >("ampCut_barrelP",{8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,
                                                   8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,
                                                   8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.});
  desc.add<std::vector<double> >("ampCut_barrelM",{8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,
                                                   8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,
                                                   8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8.});
  desc.add<std::vector<double> >("ampCut_endcapP",{12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,
                                                   12.,12.,12.,12.,12.});
  desc.add<std::vector<double> >("ampCut_endcapM",{12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,12.,
                                                   12.,12.,12.,12.,12.});
  desc.add<std::string>("phiSymBarrelDigiCollection","phiSymEcalDigisEB");
  desc.add<std::string>("phiSymEndcapDigiCollection","phiSymEcalDigisEE");
  descriptions.add("hltEcalPhiSymFilter",desc);
}


// ------------ method called to produce the data  ------------
bool 
HLTEcalPhiSymFilter::filter(edm::StreamID, edm::Event & event, const edm::EventSetup & setup) const
{
  using namespace edm;
  using namespace std;
  
  //Get ChannelStatus from DB
  edm::ESHandle<EcalChannelStatus> csHandle;
  if (! useRecoFlag_) setup.get<EcalChannelStatusRcd>().get(csHandle);
  const EcalChannelStatus& channelStatus = *csHandle; 

  //Get iRing-geometry 
  edm::ESHandle<CaloGeometry> geoHandle;
  setup.get<CaloGeometryRecord>().get(geoHandle);
  EcalRingCalibrationTools::setCaloGeometry(geoHandle.product()); 
  EcalRingCalibrationTools CalibRing;

  static const short N_RING_BARREL = EcalRingCalibrationTools::N_RING_BARREL;
  static const short N_RING_ENDCAP = EcalRingCalibrationTools::N_RING_ENDCAP;

  Handle<EBDigiCollection> barrelDigisHandle;
  Handle<EEDigiCollection> endcapDigisHandle;
  Handle<EcalUncalibratedRecHitCollection> barrelUncalibRecHitsHandle;
  Handle<EcalUncalibratedRecHitCollection> endcapUncalibRecHitsHandle;
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;

  event.getByToken(barrelDigisToken_,barrelDigisHandle);
  event.getByToken(endcapDigisToken_,endcapDigisHandle);  
  event.getByToken(barrelUncalibHitsToken_,barrelUncalibRecHitsHandle);
  event.getByToken(endcapUncalibHitsToken_,endcapUncalibRecHitsHandle);
  event.getByToken(barrelHitsToken_,barrelRecHitsHandle);
  event.getByToken(endcapHitsToken_,endcapRecHitsHandle);
 
  //Create empty output collections
  std::auto_ptr< EBDigiCollection > phiSymEBDigiCollection( new EBDigiCollection );
  std::auto_ptr< EEDigiCollection > phiSymEEDigiCollection( new EEDigiCollection );
  
  const EBDigiCollection* EBDigis = barrelDigisHandle.product();
  const EEDigiCollection* EEDigis = endcapDigisHandle.product();
  const EBRecHitCollection* EBRechits = barrelRecHitsHandle.product();
  const EERecHitCollection* EERechits = endcapRecHitsHandle.product();

  //Select interesting EcalDigis (barrel)
  EcalUncalibratedRecHitCollection::const_iterator itunb;
  for (itunb=barrelUncalibRecHitsHandle->begin(); itunb!=barrelUncalibRecHitsHandle->end(); itunb++) {
    EcalUncalibratedRecHit hit = (*itunb);
    EBDetId hitDetId = hit.id();
    uint16_t statusCode = 0; 
    if (useRecoFlag_) statusCode=(*EBRechits->find(hit.id())).recoFlag();
    else statusCode = channelStatus[itunb->id().rawId()].getStatusCode();
    int iRing = CalibRing.getRingIndex(hitDetId);
    float ampCut = 0.;
    if(hitDetId.ieta()<0) ampCut = ampCut_barlM_[iRing];
    else if(hitDetId.ieta()>0) ampCut = ampCut_barlP_[iRing-N_RING_BARREL/2];
    float amplitude = hit.amplitude();
    if( statusCode <=statusThreshold_ && amplitude > ampCut){
        phiSymEBDigiCollection->push_back((*EBDigis->find(hit.id())).id(),(*EBDigis->find(hit.id())).begin());
    }
  }
  
  //Select interesting EcalDigis (endcaps)
  EcalUncalibratedRecHitCollection::const_iterator itune;
  for (itune=endcapUncalibRecHitsHandle->begin(); itune!=endcapUncalibRecHitsHandle->end(); itune++) {
    EcalUncalibratedRecHit hit = (*itune);
    EEDetId hitDetId = hit.id();
    uint16_t statusCode = 0; 
    if (useRecoFlag_) statusCode=(*EERechits->find(hit.id())).recoFlag();
    else statusCode = channelStatus[itune->id().rawId()].getStatusCode();
    int iRing = CalibRing.getRingIndex(hitDetId);
    float ampCut = 0.;
    if(hitDetId.zside()<0) ampCut = ampCut_endcM_[iRing-N_RING_BARREL];
    else if(hitDetId.zside()>0) ampCut = ampCut_endcP_[iRing-N_RING_BARREL-N_RING_ENDCAP/2];
    float amplitude = hit.amplitude();
    if( statusCode <=statusThreshold_ && amplitude > ampCut){
        phiSymEEDigiCollection->push_back((*EEDigis->find(hit.id())).id(),(*EEDigis->find(hit.id())).begin());
    }
  }

  if ((!phiSymEBDigiCollection->size() ) && (!phiSymEEDigiCollection->size())) 
    return false;

  //Put selected information in the event
  event.put( phiSymEBDigiCollection, phiSymBarrelDigis_);
  event.put( phiSymEEDigiCollection, phiSymEndcapDigis_);
  
  return true;
}
