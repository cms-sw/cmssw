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
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

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

class testGeneralTracks : public edm::EDProducer {

public :
  explicit testGeneralTracks(const edm::ParameterSet&);
  ~testGeneralTracks();

  virtual void produce(edm::Event&, const edm::EventSetup& ) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const& ) override;
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  edm::ParameterSet particleFilter_;
  std::vector<edm::InputTag> allTracks;
  bool saveNU;
  std::vector<FSimEvent*> mySimEvent;
  std::string simModuleLabel_;
  // Histograms
  DQMStore * dbe;
  std::vector<MonitorElement*> h0;
  MonitorElement* genTracksvsEtaP;
  std::vector<MonitorElement*> TracksvsEtaP;
  std::vector<MonitorElement*> HitsvsP;
  std::vector<MonitorElement*> HitsvsEta;
  std::vector<MonitorElement*> LayersvsP;
  std::vector<MonitorElement*> LayersvsEta;

  std::vector<MonitorElement*> Num;

  std::string outputFileName;

  int totalNEvt;

  const TrackerGeometry*  theGeometry;

  int  numfast;
  int  numfull;
  int  numfastHP;
  int  numfullHP;

  reco::TrackBase::TrackQuality _trackQuality;


};

testGeneralTracks::testGeneralTracks(const edm::ParameterSet& p) :
  mySimEvent(2, static_cast<FSimEvent*>(0)),
  h0(2,static_cast<MonitorElement*>(0)),
  TracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  HitsvsP(2,static_cast<MonitorElement*>(0)),
  HitsvsEta(2,static_cast<MonitorElement*>(0)),
  LayersvsP(2,static_cast<MonitorElement*>(0)),
  LayersvsEta(2,static_cast<MonitorElement*>(0)),

  Num(2,static_cast<MonitorElement*>(0)),

  totalNEvt(0)
{
  
  // This producer produce a vector of SimTracks
  produces<edm::SimTrackContainer>();

  // Let's just initialize the SimEvent's
  particleFilter_ = p.getParameter<edm::ParameterSet>
    ( "TestParticleFilter" );   

  allTracks.push_back(p.getParameter<edm::InputTag>("Full"));
  allTracks.push_back(p.getParameter<edm::InputTag>("Fast"));
 
  // For the full sim
  mySimEvent[0] = new FSimEvent(particleFilter_);
  // For the fast sim
  mySimEvent[1] = new FSimEvent(particleFilter_);

  outputFileName = 
    p.getUntrackedParameter<std::string>("OutputFile","testGeneralTracks.root");
    
  // ... and the histograms
  dbe = edm::Service<DQMStore>().operator->();
  h0[0] = dbe->book1D("generatedEta", "Generated Eta", 300, -3., 3. );
  h0[1] = dbe->book1D("generatedMom", "Generated momentum", 100, 0., 10. );

  genTracksvsEtaP = dbe->book2D("genEtaP","Generated eta vs p",28,-2.8,2.8,100,0,10.);
  TracksvsEtaP[0] = dbe->book2D("eff0Full","Efficiency 0th Full",28,-2.8,2.8,100,0,10.);
  TracksvsEtaP[1] = dbe->book2D("eff0Fast","Efficiency 0th Fast",28,-2.8,2.8,100,0,10.);
  HitsvsP[0]      = dbe->book2D("Hits0PFull","Hits vs P 0th Full",100,0.,10.,30,0,30.);
  HitsvsP[1]      = dbe->book2D("Hits0PFast","Hits vs P 0th Fast",100,0.,10.,30,0,30.);
  HitsvsEta[0]    = dbe->book2D("Hits0EtaFull","Hits vs Eta 0th Full",28,-2.8,2.8,30,0,30.);
  HitsvsEta[1]    = dbe->book2D("Hits0EtaFast","Hits vs Eta 0th Fast",28,-2.8,2.8,30,0,30.);
  LayersvsP[0]    = dbe->book2D("Layers0PFull","Layers vs P 0th Full",100,0.,10.,30,0,30.);
  LayersvsP[1]    = dbe->book2D("Layers0PFast","Layers vs P 0th Fast",100,0.,10.,30,0,30.);
  LayersvsEta[0]  = dbe->book2D("Layers0EtaFull","Layers vs Eta 0th Full",28,-2.8,2.8,30,0,30.);
  LayersvsEta[1]  = dbe->book2D("Layers0EtaFast","Layers vs Eta 0th Fast",28,-2.8,2.8,30,0,30.);

}

testGeneralTracks::~testGeneralTracks()
{

  std::cout << "\t\t Number of Tracks " << std::endl;
  std::cout << "\tFULL\t" <<  numfull << "\t HP= " << numfullHP << std::endl;
  std::cout << "\tFAST\t" <<  numfast << "\t HP= " << numfastHP << std::endl;

  dbe->save(outputFileName);

}

void testGeneralTracks::beginRun(edm::Run const&, edm::EventSetup const& es)
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

  numfast = numfull = 0;
  numfastHP = numfullHP = 0;
  _trackQuality = reco::TrackBase::qualityByName("highPurity");

}

void
testGeneralTracks::produce(edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  
  ++totalNEvt;
  
  //  std::cout << " >>>>>>>>> Analizying Event " << totalNEvt << "<<<<<<< " << std::endl; 
  
  if ( totalNEvt/1000*1000 == totalNEvt ) 
    std::cout << "Number of event analysed "
	      << totalNEvt << std::endl; 
  
  std::auto_ptr<edm::SimTrackContainer> nuclSimTracks(new edm::SimTrackContainer);
  
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
  //  std::cout << " PArticle list: Pt = "  <<  pGen << " , eta = " << etaGen << std::endl;
  
  std::vector<bool> firstSeed(2,static_cast<bool>(false));
  std::vector<bool> secondSeed(2,static_cast<bool>(false));
  std::vector<TrajectorySeed::range> theRecHitRange(2);
  
  for ( unsigned ievt=0; ievt<2; ++ievt ) {
    
    edm::Handle<reco::TrackCollection> tkRef0;
    iEvent.getByLabel(allTracks[ievt],tkRef0);    
    std::vector<const reco::TrackCollection*> tkColl;
    tkColl.push_back(tkRef0.product());
    
    
    reco::TrackCollection::const_iterator itk0 = tkColl[0]->begin();
    reco::TrackCollection::const_iterator itk0_e = tkColl[0]->end();
    for(;itk0!=itk0_e;++itk0){
      //std::cout << "quality " << itk0->quality(_trackQuality) << std::endl;
      if(!(itk0->quality(_trackQuality)) ) {
	//std::cout << "evt " << totalNEvt << "\tTRACK REMOVED" << std::endl;
	continue;
      } 
      if(ievt==0) numfullHP++;
      if(ievt==1) numfastHP++;
     TracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
     HitsvsEta[ievt]->Fill(etaGen,itk0->found(),1.);
     HitsvsP[ievt]->Fill(pGen,itk0->found(),1.);
     LayersvsEta[ievt]->Fill(etaGen,itk0->hitPattern().trackerLayersWithMeasurement(),1.);
     LayersvsP[ievt]->Fill(pGen,itk0->hitPattern().trackerLayersWithMeasurement(),1.);
    }

    //    std::cout << "\t\t Number of Tracks " << std::endl;
    if(ievt ==0){
      numfull +=  tkColl[0]->size();       
      //      std::cout << "\tFULL\t" << tkColl[0]->size() << "\t" << numfull << "\t" << numfullHP << std::endl;
    } else if (ievt ==1){
      numfast +=  tkColl[0]->size();
      // std::cout << "\tFAST\t" << tkColl[0]->size() << "\t" << numfast << "\t" << numfastHP << std::endl;
    }


  }
}

//define this as a plug-in

DEFINE_FWK_MODULE(testGeneralTracks);
