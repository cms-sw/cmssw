// user include files
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"

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

  std::string outputFileName;

  int totalNEvt;

};

testTrackingIterations::testTrackingIterations(const edm::ParameterSet& p) :
  mySimEvent(2, static_cast<FSimEvent*>(0)),
  h0(2,static_cast<MonitorElement*>(0)),
  firstTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  first1TracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  first2TracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  secondTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  thirdTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
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
    if ( tkColl[0]->size() == 1 ) firstTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[0]->size());
    if ( tkColl[1]->size() == 1 ) secondTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[1]->size());
    if ( tkColl[2]->size() == 1 ) thirdTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[2]->size());

    // Split 1st collection in two (triplets, then pairs)
    reco::TrackCollection::const_iterator itk = tkColl[0]->begin();
    if ( tkColl[0]->size()==1 ) {
      const TrajectorySeed* seed = itk->seedRef().get();
      double NumberofhitsSeed = seed->recHits().second-seed->recHits().first;
      if ( NumberofhitsSeed == 3 ) 
	first1TracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
      else
	first2TracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
    }
  }

}

//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testTrackingIterations);
