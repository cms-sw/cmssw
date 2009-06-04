// user include files
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/Tracking/interface/TrackerRecHit.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <vector>
#include <string>
//#include "TFile.h"
//#include "TTree.h"
//#include "TProcessID.h"

class testTrackingIterations : public edm::EDProducer {

public :
  explicit testTrackingIterations(const edm::ParameterSet&);
  ~testTrackingIterations();

  virtual void produce(edm::Event&, const edm::EventSetup& );
  virtual void beginJob(const edm::EventSetup & c);
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  edm::ParameterSet particleFilter_;
  std::vector<edm::InputTag> firstTracks;
  std::vector<edm::InputTag> secondTracks;
  std::vector<edm::InputTag> thirdTracks;
  bool saveNU;
  std::vector<FSimEvent*> mySimEvent;
  std::string simModuleLabel_;
  // Histograms
  DQMStore * dbe;
  std::vector<MonitorElement*> h0;
  MonitorElement* genTracksvsEtaP;
  std::vector<MonitorElement*> firstTracksvsEtaP;
  std::vector<MonitorElement*> first1TracksvsEtaP;
  std::vector<MonitorElement*> first2TracksvsEtaP;
  std::vector<MonitorElement*> secondTracksvsEtaP;
  std::vector<MonitorElement*> thirdTracksvsEtaP;
  std::vector<MonitorElement*> firstHitsvsP;
  std::vector<MonitorElement*> secondHitsvsP;
  std::vector<MonitorElement*> thirdHitsvsP;
  std::vector<MonitorElement*> firstHitsvsEta;
  std::vector<MonitorElement*> secondHitsvsEta;
  std::vector<MonitorElement*> thirdHitsvsEta;
  std::vector<MonitorElement*> firstLayersvsP;
  std::vector<MonitorElement*> secondLayersvsP;
  std::vector<MonitorElement*> thirdLayersvsP;
  std::vector<MonitorElement*> firstLayersvsEta;
  std::vector<MonitorElement*> secondLayersvsEta;
  std::vector<MonitorElement*> thirdLayersvsEta;

  std::string outputFileName;

  int totalNEvt;

  const TrackerGeometry*  theGeometry;

};

testTrackingIterations::testTrackingIterations(const edm::ParameterSet& p) :
  mySimEvent(2, static_cast<FSimEvent*>(0)),
  h0(2,static_cast<MonitorElement*>(0)),
  firstTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  first1TracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  first2TracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  secondTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  thirdTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  firstHitsvsP(2,static_cast<MonitorElement*>(0)),
  secondHitsvsP(2,static_cast<MonitorElement*>(0)),
  thirdHitsvsP(2,static_cast<MonitorElement*>(0)),
  firstHitsvsEta(2,static_cast<MonitorElement*>(0)),
  secondHitsvsEta(2,static_cast<MonitorElement*>(0)),
  thirdHitsvsEta(2,static_cast<MonitorElement*>(0)),
  firstLayersvsP(2,static_cast<MonitorElement*>(0)),
  secondLayersvsP(2,static_cast<MonitorElement*>(0)),
  thirdLayersvsP(2,static_cast<MonitorElement*>(0)),
  firstLayersvsEta(2,static_cast<MonitorElement*>(0)),
  secondLayersvsEta(2,static_cast<MonitorElement*>(0)),
  thirdLayersvsEta(2,static_cast<MonitorElement*>(0)),
  totalNEvt(0)
{
  
  // This producer produce a vector of SimTracks
  produces<edm::SimTrackContainer>();

  // Let's just initialize the SimEvent's
  particleFilter_ = p.getParameter<edm::ParameterSet>
    ( "TestParticleFilter" );   

  firstTracks.push_back(p.getParameter<edm::InputTag>("firstFull"));
  firstTracks.push_back(p.getParameter<edm::InputTag>("firstFast"));
  secondTracks.push_back(p.getParameter<edm::InputTag>("secondFull"));
  secondTracks.push_back(p.getParameter<edm::InputTag>("secondFast"));
  thirdTracks.push_back(p.getParameter<edm::InputTag>("thirdFull"));
  thirdTracks.push_back(p.getParameter<edm::InputTag>("thirdFast"));

  // For the full sim
  mySimEvent[0] = new FSimEvent(particleFilter_);
  // For the fast sim
  mySimEvent[1] = new FSimEvent(particleFilter_);

  outputFileName = 
    p.getUntrackedParameter<std::string>("OutputFile","testTrackingIterations.root");
    
  // ... and the histograms
  dbe = edm::Service<DQMStore>().operator->();
  h0[0] = dbe->book1D("generatedEta", "Generated Eta", 300, -3., 3. );
  h0[1] = dbe->book1D("generatedMom", "Generated momentum", 100, 0., 10. );

  genTracksvsEtaP = dbe->book2D("genEtaP","Generated eta vs p",28,-2.8,2.8,100,0,10.);
  firstTracksvsEtaP[0] = dbe->book2D("eff1Full","Efficiency 1st Full",28,-2.8,2.8,100,0,10.);
  firstTracksvsEtaP[1] = dbe->book2D("eff1Fast","Efficiency 1st Fast",28,-2.8,2.8,100,0,10.);
  first1TracksvsEtaP[0] = dbe->book2D("eff1Full1","Efficiency 1st Full 1",28,-2.8,2.8,100,0,10.);
  first1TracksvsEtaP[1] = dbe->book2D("eff1Fast1","Efficiency 1st Fast 1",28,-2.8,2.8,100,0,10.);
  first2TracksvsEtaP[0] = dbe->book2D("eff1Full2","Efficiency 1st Full 2",28,-2.8,2.8,100,0,10.);
  first2TracksvsEtaP[1] = dbe->book2D("eff1Fast2","Efficiency 1st Fast 2",28,-2.8,2.8,100,0,10.);
  secondTracksvsEtaP[0] = dbe->book2D("eff2Full","Efficiency 2nd Full",28,-2.8,2.8,100,0,10.);
  secondTracksvsEtaP[1] = dbe->book2D("eff2Fast","Efficiency 2nd Fast",28,-2.8,2.8,100,0,10.);
  thirdTracksvsEtaP[0] = dbe->book2D("eff3Full","Efficiency 3rd Full",28,-2.8,2.8,100,0,10.);
  thirdTracksvsEtaP[1] = dbe->book2D("eff3Fast","Efficiency 3rd Fast",28,-2.8,2.8,100,0,10.);
  firstHitsvsP[0] = dbe->book2D("Hits1PFull","Hits vs P 1st Full",100,0.,10.,30,0,30.);
  firstHitsvsP[1] = dbe->book2D("Hits1PFast","Hits vs P 1st Fast",100,0.,10.,30,0,30.);
  secondHitsvsP[0] = dbe->book2D("Hits2PFull","Hits vs P 2nd Full",100,0.,10.,30,0,30.);
  secondHitsvsP[1] = dbe->book2D("Hits2PFast","Hits vs P 2nd Fast",100,0.,10.,30,0,30.);
  thirdHitsvsP[0] = dbe->book2D("Hits3PFull","Hits vs P 3rd Full",100,0.,10.,30,0,30.);
  thirdHitsvsP[1] = dbe->book2D("Hits3PFast","Hits vs P 3rd Fast",100,0.,10.,30,0,30.);
  firstHitsvsEta[0] = dbe->book2D("Hits1EtaFull","Hits vs Eta 1st Full",28,-2.8,2.8,30,0,30.);
  firstHitsvsEta[1] = dbe->book2D("Hits1EtaFast","Hits vs Eta 1st Fast",28,-2.8,2.8,30,0,30.);
  secondHitsvsEta[0] = dbe->book2D("Hits2EtaFull","Hits vs Eta 2nd Full",28,-2.8,2.8,30,0,30.);
  secondHitsvsEta[1] = dbe->book2D("Hits2EtaFast","Hits vs Eta 2nd Fast",28,-2.8,2.8,30,0,30.);
  thirdHitsvsEta[0] = dbe->book2D("Hits3EtaFull","Hits vs Eta 3rd Full",28,-2.8,2.8,30,0,30.);
  thirdHitsvsEta[1] = dbe->book2D("Hits3EtaFast","Hits vs Eta 3rd Fast",28,-2.8,2.8,30,0,30.);
  firstLayersvsP[0] = dbe->book2D("Layers1PFull","Layers vs P 1st Full",100,0.,10.,30,0,30.);
  firstLayersvsP[1] = dbe->book2D("Layers1PFast","Layers vs P 1st Fast",100,0.,10.,30,0,30.);
  secondLayersvsP[0] = dbe->book2D("Layers2PFull","Layers vs P 2nd Full",100,0.,10.,30,0,30.);
  secondLayersvsP[1] = dbe->book2D("Layers2PFast","Layers vs P 2nd Fast",100,0.,10.,30,0,30.);
  thirdLayersvsP[0] = dbe->book2D("Layers3PFull","Layers vs P 3rd Full",100,0.,10.,30,0,30.);
  thirdLayersvsP[1] = dbe->book2D("Layers3PFast","Layers vs P 3rd Fast",100,0.,10.,30,0,30.);
  firstLayersvsEta[0] = dbe->book2D("Layers1EtaFull","Layers vs Eta 1st Full",28,-2.8,2.8,30,0,30.);
  firstLayersvsEta[1] = dbe->book2D("Layers1EtaFast","Layers vs Eta 1st Fast",28,-2.8,2.8,30,0,30.);
  secondLayersvsEta[0] = dbe->book2D("Layers2EtaFull","Layers vs Eta 2nd Full",28,-2.8,2.8,30,0,30.);
  secondLayersvsEta[1] = dbe->book2D("Layers2EtaFast","Layers vs Eta 2nd Fast",28,-2.8,2.8,30,0,30.);
  thirdLayersvsEta[0] = dbe->book2D("Layers3EtaFull","Layers vs Eta 3rd Full",28,-2.8,2.8,30,0,30.);
  thirdLayersvsEta[1] = dbe->book2D("Layers3EtaFast","Layers vs Eta 3rd Fast",28,-2.8,2.8,30,0,30.);
								
}

testTrackingIterations::~testTrackingIterations()
{

  dbe->save(outputFileName);

}

void testTrackingIterations::beginJob(const edm::EventSetup & es)
{
  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  if ( !ParticleTable::instance() ) ParticleTable::instance(&(*pdt));
  mySimEvent[0]->initializePdt(&(*pdt));
  mySimEvent[1]->initializePdt(&(*pdt));

  edm::ESHandle<TrackerGeometry>        geometry;
  es.get<TrackerDigiGeometryRecord>().get(geometry);
  theGeometry = &(*geometry);

}

void
testTrackingIterations::produce(edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  ++totalNEvt;
  if ( totalNEvt/1000*1000 == totalNEvt ) 
    std::cout << "Number of event analysed "
	      << totalNEvt << std::endl; 

  std::auto_ptr<edm::SimTrackContainer> nuclSimTracks(new edm::SimTrackContainer);


  /*
  // Fill sim events.
  //  std::cout << "Fill full event " << std::endl;
  edm::Handle<std::vector<SimTrack> > fullSimTracks;
  iEvent.getByLabel("g4SimHits",fullSimTracks);
  edm::Handle<std::vector<SimVertex> > fullSimVertices;
  iEvent.getByLabel("g4SimHits",fullSimVertices);
  mySimEvent[0]->fill( *fullSimTracks, *fullSimVertices );
  */
  
  edm::Handle<std::vector<SimTrack> > fastSimTracks;
  iEvent.getByLabel("famosSimHits",fastSimTracks);
  edm::Handle<std::vector<SimVertex> > fastSimVertices;
  iEvent.getByLabel("famosSimHits",fastSimVertices);
  mySimEvent[1]->fill( *fastSimTracks, *fastSimVertices );

  if ( !mySimEvent[1]->nVertices() ) return;
  if ( !mySimEvent[1]->nTracks() ) return;
  const FSimTrack& thePion = mySimEvent[1]->track(0);
  
  double etaGen = thePion.momentum().Eta();
  double pGen = std::sqrt(thePion.momentum().Vect().Perp2());
  if ( pGen < 0.2 ) return;

  h0[0]->Fill(pGen);
  h0[1]->Fill(etaGen);
  genTracksvsEtaP->Fill(etaGen,pGen,1.);

  std::vector<bool> firstSeed(2,static_cast<bool>(false));
  std::vector<bool> secondSeed(2,static_cast<bool>(false));
  std::vector<TrajectorySeed::range> theRecHitRange(2);

  for ( unsigned ievt=0; ievt<2; ++ievt ) {

    edm::Handle<reco::TrackCollection> tkRef1;
    edm::Handle<reco::TrackCollection> tkRef2;
    edm::Handle<reco::TrackCollection> tkRef3;
    iEvent.getByLabel(firstTracks[ievt],tkRef1);    
    iEvent.getByLabel(secondTracks[ievt],tkRef2);    
    iEvent.getByLabel(thirdTracks[ievt],tkRef3);   
    std::vector<const reco::TrackCollection*> tkColl;
    tkColl.push_back(tkRef1.product());
    tkColl.push_back(tkRef2.product());
    tkColl.push_back(tkRef3.product());
    // if ( tkColl[0]+tkColl[1]+tkColl[2] != 1 ) continue;
    if ( tkColl[0]->size() == 1 ) { 
      firstTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[0]->size());
      reco::TrackCollection::const_iterator itk = tkColl[0]->begin();
      firstHitsvsEta[ievt]->Fill(etaGen,itk->found(),1.);
      firstHitsvsP[ievt]->Fill(pGen,itk->found(),1.);
      firstLayersvsEta[ievt]->Fill(etaGen,itk->hitPattern().trackerLayersWithMeasurement(),1.);
      firstLayersvsP[ievt]->Fill(pGen,itk->hitPattern().trackerLayersWithMeasurement(),1.);
    }
    if ( tkColl[1]->size() == 1 ) { 
      secondTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[1]->size());
      reco::TrackCollection::const_iterator itk = tkColl[1]->begin();
      secondHitsvsEta[ievt]->Fill(etaGen,itk->found(),1.);    
      secondHitsvsP[ievt]->Fill(pGen,itk->found(),1.);
      secondLayersvsEta[ievt]->Fill(etaGen,itk->hitPattern().trackerLayersWithMeasurement(),1.);    
      secondLayersvsP[ievt]->Fill(pGen,itk->hitPattern().trackerLayersWithMeasurement(),1.);
    }
    if ( tkColl[2]->size() == 1 ) { 
      thirdTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[2]->size());
      reco::TrackCollection::const_iterator itk = tkColl[2]->begin();
      thirdHitsvsEta[ievt]->Fill(etaGen,itk->found(),1.);
      thirdHitsvsP[ievt]->Fill(pGen,itk->found(),1.);
      thirdLayersvsEta[ievt]->Fill(etaGen,itk->hitPattern().trackerLayersWithMeasurement(),1.);
      thirdLayersvsP[ievt]->Fill(pGen,itk->hitPattern().trackerLayersWithMeasurement(),1.);
    }

    // Split 1st collection in two (triplets, then pairs)
    if ( tkColl[0]->size()>=1 ) {
      reco::TrackCollection::const_iterator itk = tkColl[0]->begin();
      const TrajectorySeed* seed = itk->seedRef().get();
      // double pNumberofhitsSeed = seed->recHits().second-seed->recHits().first;
      double NumberofhitsSeed = seed->nHits();
      if ( NumberofhitsSeed == 3 ) { 
	first1TracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
	firstSeed[ievt] = true;
	theRecHitRange[ievt] = seed->recHits();
      } else { 
	first2TracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
	secondSeed[ievt] = true;
      }
    }

  }

  /*
  if ( firstSeed[0] != firstSeed[1] && secondSeed[0] != secondSeed[1] ) { 
    std::cout << "First  Seed found in full / fast ? " << firstSeed[0] << " " << firstSeed[1] << std::endl
	      << "Second Seed found in full / fast ? " << secondSeed[0] << " " << secondSeed[1] << std::endl
	      << "eta/p = " << etaGen << ", " << pGen << std::endl;
    TrajectorySeed::const_iterator firstHit, lastHit;
    if ( firstSeed[0] ) { 
      firstHit = theRecHitRange[0].first;
      lastHit = theRecHitRange[0].second;
    } else {
      firstHit = theRecHitRange[1].first;
      lastHit = theRecHitRange[1].second;
    }

    unsigned hit = 0;
    const GeomDet* theGeomDet;
    unsigned int theSubDetId; 
    unsigned int theLayerNumber;
    unsigned int theRingNumber;
    bool forward;
    for ( ; firstHit != lastHit; ++firstHit ) { 
      const DetId& theDetId = firstHit->geographicalId();
      theGeomDet = theGeometry->idToDet(theDetId);
      theSubDetId = theDetId.subdetId(); 
      if ( theSubDetId == StripSubdetector::TIB) { 
	TIBDetId tibid(theDetId.rawId()); 
	theLayerNumber = tibid.layer();
	forward = false;
      } else if ( theSubDetId ==  StripSubdetector::TOB ) { 
	TOBDetId tobid(theDetId.rawId()); 
	theLayerNumber = tobid.layer();
	forward = false;
      } else if ( theSubDetId ==  StripSubdetector::TID) { 
	TIDDetId tidid(theDetId.rawId());
	theLayerNumber = tidid.wheel();
	theRingNumber = tidid.ring();
	forward = true;
      } else if ( theSubDetId ==  StripSubdetector::TEC ) { 
	TECDetId tecid(theDetId.rawId()); 
	theLayerNumber = tecid.wheel(); 
	theRingNumber = tecid.ring();
	forward = true;
      } else if ( theSubDetId ==  PixelSubdetector::PixelBarrel ) { 
	PXBDetId pxbid(theDetId.rawId()); 
	theLayerNumber = pxbid.layer(); 
	forward = false;
      } else if ( theSubDetId ==  PixelSubdetector::PixelEndcap ) { 
	PXFDetId pxfid(theDetId.rawId()); 
	theLayerNumber = pxfid.disk();  
	forward = true;
      }


      std::cout << "Hit " << hit++ << " Subdet " << theSubDetId << ", Layer " << theLayerNumber << std::endl; 
    }
  }
  */

}

//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testTrackingIterations);
