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
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h" 
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

#define DEBUG false

class testTrackingIterations : public edm::EDProducer {

public :
  explicit testTrackingIterations(const edm::ParameterSet&);
  ~testTrackingIterations();

  virtual void produce(edm::Event&, const edm::EventSetup& );
  virtual void beginRun(edm::Run&, edm::EventSetup const& );
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  edm::ParameterSet particleFilter_;
  std::vector<edm::InputTag> zeroTracks;
  std::vector<edm::InputTag> firstTracks;
  std::vector<edm::InputTag> secondTracks;
  std::vector<edm::InputTag> thirdTracks;
  std::vector<edm::InputTag> fourthTracks;
  std::vector<edm::InputTag> fifthTracks;
  bool saveNU;
  std::vector<FSimEvent*> mySimEvent;
  std::string simModuleLabel_;
  // Histograms
  DQMStore * dbe;
  std::vector<MonitorElement*> h0;
  MonitorElement* genTracksvsEtaP;
  std::vector<MonitorElement*> SimTracksvsEtaP;
  std::vector<MonitorElement*> zeroTracksvsEtaP;
  std::vector<MonitorElement*> firstTracksvsEtaP;
  std::vector<MonitorElement*> first1TracksvsEtaP;
  std::vector<MonitorElement*> first2TracksvsEtaP;
  std::vector<MonitorElement*> secondTracksvsEtaP;
  std::vector<MonitorElement*> thirdTracksvsEtaP;
  std::vector<MonitorElement*> fourthTracksvsEtaP;
  std::vector<MonitorElement*> fifthTracksvsEtaP;
  std::vector<MonitorElement*> zeroHitsvsP;
  std::vector<MonitorElement*> firstHitsvsP;
  std::vector<MonitorElement*> secondHitsvsP;
  std::vector<MonitorElement*> thirdHitsvsP;
  std::vector<MonitorElement*> fourthHitsvsP;
  std::vector<MonitorElement*> fifthHitsvsP;
  std::vector<MonitorElement*> zeroHitsvsEta;
  std::vector<MonitorElement*> firstHitsvsEta;
  std::vector<MonitorElement*> secondHitsvsEta;
  std::vector<MonitorElement*> thirdHitsvsEta;
  std::vector<MonitorElement*> fourthHitsvsEta;
  std::vector<MonitorElement*> fifthHitsvsEta;
  std::vector<MonitorElement*> zeroLayersvsP;
  std::vector<MonitorElement*> firstLayersvsP;
  std::vector<MonitorElement*> secondLayersvsP;
  std::vector<MonitorElement*> thirdLayersvsP;
  std::vector<MonitorElement*> fourthLayersvsP;
  std::vector<MonitorElement*> fifthLayersvsP;
  std::vector<MonitorElement*> zeroLayersvsEta;
  std::vector<MonitorElement*> firstLayersvsEta;
  std::vector<MonitorElement*> secondLayersvsEta;
  std::vector<MonitorElement*> thirdLayersvsEta;
  std::vector<MonitorElement*> fourthLayersvsEta;
  std::vector<MonitorElement*> fifthLayersvsEta;
  std::vector<MonitorElement*> thirdSeedvsP;
  std::vector<MonitorElement*> thirdSeedvsEta;
  std::vector<MonitorElement*> fifthSeedvsP;
  std::vector<MonitorElement*> fifthSeedvsEta;

  std::vector<MonitorElement*> zeroNum;
  std::vector<MonitorElement*> firstNum;
  std::vector<MonitorElement*> secondNum;
  std::vector<MonitorElement*> thirdNum;
  std::vector<MonitorElement*> fourthNum;
  std::vector<MonitorElement*> fifthNum;

  std::string outputFileName;
  std::string TheSample(unsigned ievt);

  int totalNEvt;

  const TrackerGeometry*  theGeometry;

  int  num0fast,num1fast, num2fast, num3fast, num4fast, num5fast;
  int  num0full, num1full, num2full, num3full, num4full, num5full;
  float  MySeedType3,MySeedType5;

};


std::string testTrackingIterations::TheSample(unsigned ievt)
{
  if (ievt==0) {
    return "FULL";
  }
  else if (ievt==1) {
    return "FAST";
  }
  else {
    return "UNKW";
  }

}

testTrackingIterations::testTrackingIterations(const edm::ParameterSet& p) :

  mySimEvent(2, static_cast<FSimEvent*>(0)),
  h0(2,static_cast<MonitorElement*>(0)),
  SimTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  zeroTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  firstTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  first1TracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  first2TracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  secondTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  thirdTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  fourthTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  fifthTracksvsEtaP(2,static_cast<MonitorElement*>(0)),
  zeroHitsvsP(2,static_cast<MonitorElement*>(0)),
  firstHitsvsP(2,static_cast<MonitorElement*>(0)),
  secondHitsvsP(2,static_cast<MonitorElement*>(0)),
  thirdHitsvsP(2,static_cast<MonitorElement*>(0)),
  fourthHitsvsP(2,static_cast<MonitorElement*>(0)),
  fifthHitsvsP(2,static_cast<MonitorElement*>(0)),
  zeroHitsvsEta(2,static_cast<MonitorElement*>(0)),
  firstHitsvsEta(2,static_cast<MonitorElement*>(0)),
  secondHitsvsEta(2,static_cast<MonitorElement*>(0)),
  thirdHitsvsEta(2,static_cast<MonitorElement*>(0)),
  fourthHitsvsEta(2,static_cast<MonitorElement*>(0)),
  fifthHitsvsEta(2,static_cast<MonitorElement*>(0)),
  zeroLayersvsP(2,static_cast<MonitorElement*>(0)),
  firstLayersvsP(2,static_cast<MonitorElement*>(0)),
  secondLayersvsP(2,static_cast<MonitorElement*>(0)),
  thirdLayersvsP(2,static_cast<MonitorElement*>(0)),
  fourthLayersvsP(2,static_cast<MonitorElement*>(0)),
  fifthLayersvsP(2,static_cast<MonitorElement*>(0)),
  zeroLayersvsEta(2,static_cast<MonitorElement*>(0)),
  firstLayersvsEta(2,static_cast<MonitorElement*>(0)),
  secondLayersvsEta(2,static_cast<MonitorElement*>(0)),
  thirdLayersvsEta(2,static_cast<MonitorElement*>(0)),
  fourthLayersvsEta(2,static_cast<MonitorElement*>(0)),
  fifthLayersvsEta(2,static_cast<MonitorElement*>(0)),
  thirdSeedvsP(2,static_cast<MonitorElement*>(0)),
  thirdSeedvsEta(2,static_cast<MonitorElement*>(0)),
  fifthSeedvsP(2,static_cast<MonitorElement*>(0)),
  fifthSeedvsEta(2,static_cast<MonitorElement*>(0)),

  zeroNum(2,static_cast<MonitorElement*>(0)),
  firstNum(2,static_cast<MonitorElement*>(0)),
  secondNum(2,static_cast<MonitorElement*>(0)),
  thirdNum(2,static_cast<MonitorElement*>(0)),
  fourthNum(2,static_cast<MonitorElement*>(0)),
  fifthNum(2,static_cast<MonitorElement*>(0)),

  totalNEvt(0)
{
  
  // This producer produce a vector of SimTracks
  produces<edm::SimTrackContainer>();

  // Let's just initialize the SimEvent's
  particleFilter_ = p.getParameter<edm::ParameterSet>
    ( "TestParticleFilter" );   

  zeroTracks.push_back(p.getParameter<edm::InputTag>("zeroFull"));
  zeroTracks.push_back(p.getParameter<edm::InputTag>("zeroFast"));
  firstTracks.push_back(p.getParameter<edm::InputTag>("firstFull"));
  firstTracks.push_back(p.getParameter<edm::InputTag>("firstFast"));
  secondTracks.push_back(p.getParameter<edm::InputTag>("secondFull"));
  secondTracks.push_back(p.getParameter<edm::InputTag>("secondFast"));
  thirdTracks.push_back(p.getParameter<edm::InputTag>("thirdFull"));
  thirdTracks.push_back(p.getParameter<edm::InputTag>("thirdFast"));
  fourthTracks.push_back(p.getParameter<edm::InputTag>("fourthFull"));
  fourthTracks.push_back(p.getParameter<edm::InputTag>("fourthFast"));
  fifthTracks.push_back(p.getParameter<edm::InputTag>("fifthFull"));
  fifthTracks.push_back(p.getParameter<edm::InputTag>("fifthFast"));

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
  SimTracksvsEtaP[0] = dbe->book2D("SimFull","SimTrack Full",28,-2.8,2.8,100,0,10.);
  SimTracksvsEtaP[1] = dbe->book2D("SimFast","SimTrack Fast",28,-2.8,2.8,100,0,10.);
  genTracksvsEtaP = dbe->book2D("genEtaP","Generated eta vs p",28,-2.8,2.8,100,0,10.);
  zeroTracksvsEtaP[0] = dbe->book2D("eff0Full","Efficiency 0th Full",28,-2.8,2.8,100,0,10.);
  zeroTracksvsEtaP[1] = dbe->book2D("eff0Fast","Efficiency 0th Fast",28,-2.8,2.8,100,0,10.);
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
  fourthTracksvsEtaP[0] = dbe->book2D("eff4Full","Efficiency 4th Full",28,-2.8,2.8,100,0,10.);
  fourthTracksvsEtaP[1] = dbe->book2D("eff4Fast","Efficiency 4th Fast",28,-2.8,2.8,100,0,10.);
  fifthTracksvsEtaP[0] = dbe->book2D("eff5Full","Efficiency 5th Full",28,-2.8,2.8,100,0,10.);
  fifthTracksvsEtaP[1] = dbe->book2D("eff5Fast","Efficiency 5th Fast",28,-2.8,2.8,100,0,10.);

  zeroHitsvsP[0] = dbe->book2D("Hits0PFull","Hits vs P 0th Full",100,0.,10.,30,0,30.);
  zeroHitsvsP[1] = dbe->book2D("Hits0PFast","Hits vs P 0th Fast",100,0.,10.,30,0,30.);
  firstHitsvsP[0] = dbe->book2D("Hits1PFull","Hits vs P 1st Full",100,0.,10.,30,0,30.);
  firstHitsvsP[1] = dbe->book2D("Hits1PFast","Hits vs P 1st Fast",100,0.,10.,30,0,30.);
  secondHitsvsP[0] = dbe->book2D("Hits2PFull","Hits vs P 2nd Full",100,0.,10.,30,0,30.);
  secondHitsvsP[1] = dbe->book2D("Hits2PFast","Hits vs P 2nd Fast",100,0.,10.,30,0,30.);
  thirdHitsvsP[0] = dbe->book2D("Hits3PFull","Hits vs P 3rd Full",100,0.,10.,30,0,30.);
  thirdHitsvsP[1] = dbe->book2D("Hits3PFast","Hits vs P 3rd Fast",100,0.,10.,30,0,30.);
  fourthHitsvsP[0] = dbe->book2D("Hits4PFull","Hits vs P 4th Full",100,0.,10.,30,0,30.);
  fourthHitsvsP[1] = dbe->book2D("Hits4PFast","Hits vs P 4th Fast",100,0.,10.,30,0,30.);
  fifthHitsvsP[0] = dbe->book2D("Hits5PFull","Hits vs P 5th Full",100,0.,10.,30,0,30.);
  fifthHitsvsP[1] = dbe->book2D("Hits5PFast","Hits vs P 5th Fast",100,0.,10.,30,0,30.);
  zeroHitsvsEta[0] = dbe->book2D("Hits0EtaFull","Hits vs Eta 0th Full",28,-2.8,2.8,30,0,30.);
  zeroHitsvsEta[1] = dbe->book2D("Hits0EtaFast","Hits vs Eta 0th Fast",28,-2.8,2.8,30,0,30.);
  firstHitsvsEta[0] = dbe->book2D("Hits1EtaFull","Hits vs Eta 1st Full",28,-2.8,2.8,30,0,30.);
  firstHitsvsEta[1] = dbe->book2D("Hits1EtaFast","Hits vs Eta 1st Fast",28,-2.8,2.8,30,0,30.);
  secondHitsvsEta[0] = dbe->book2D("Hits2EtaFull","Hits vs Eta 2nd Full",28,-2.8,2.8,30,0,30.);
  secondHitsvsEta[1] = dbe->book2D("Hits2EtaFast","Hits vs Eta 2nd Fast",28,-2.8,2.8,30,0,30.);
  thirdHitsvsEta[0] = dbe->book2D("Hits3EtaFull","Hits vs Eta 3rd Full",28,-2.8,2.8,30,0,30.);
  thirdHitsvsEta[1] = dbe->book2D("Hits3EtaFast","Hits vs Eta 3rd Fast",28,-2.8,2.8,30,0,30.);
  fourthHitsvsEta[0] = dbe->book2D("Hits4EtaFull","Hits vs Eta 4th Full",28,-2.8,2.8,30,0,30.);
  fourthHitsvsEta[1] = dbe->book2D("Hits4EtaFast","Hits vs Eta 4th Fast",28,-2.8,2.8,30,0,30.);
  fifthHitsvsEta[0] = dbe->book2D("Hits5EtaFull","Hits vs Eta 5th Full",28,-2.8,2.8,30,0,30.);
  fifthHitsvsEta[1] = dbe->book2D("Hits5EtaFast","Hits vs Eta 5th Fast",28,-2.8,2.8,30,0,30.);
  zeroLayersvsP[0] = dbe->book2D("Layers0PFull","Layers vs P 0th Full",100,0.,10.,30,0,30.);
  zeroLayersvsP[1] = dbe->book2D("Layers0PFast","Layers vs P 0th Fast",100,0.,10.,30,0,30.);
  firstLayersvsP[0] = dbe->book2D("Layers1PFull","Layers vs P 1st Full",100,0.,10.,30,0,30.);
  firstLayersvsP[1] = dbe->book2D("Layers1PFast","Layers vs P 1st Fast",100,0.,10.,30,0,30.);
  secondLayersvsP[0] = dbe->book2D("Layers2PFull","Layers vs P 2nd Full",100,0.,10.,30,0,30.);
  secondLayersvsP[1] = dbe->book2D("Layers2PFast","Layers vs P 2nd Fast",100,0.,10.,30,0,30.);
  thirdLayersvsP[0] = dbe->book2D("Layers3PFull","Layers vs P 3rd Full",100,0.,10.,30,0,30.);
  thirdLayersvsP[1] = dbe->book2D("Layers3PFast","Layers vs P 3rd Fast",100,0.,10.,30,0,30.);
  fourthLayersvsP[0] = dbe->book2D("Layers4PFull","Layers vs P 4th Full",100,0.,10.,30,0,30.);
  fourthLayersvsP[1] = dbe->book2D("Layers4PFast","Layers vs P 4th Fast",100,0.,10.,30,0,30.);
  fifthLayersvsP[0] = dbe->book2D("Layers5PFull","Layers vs P 5th Full",100,0.,10.,30,0,30.);
  fifthLayersvsP[1] = dbe->book2D("Layers5PFast","Layers vs P 5th Fast",100,0.,10.,30,0,30.);
  zeroLayersvsEta[0] = dbe->book2D("Layers0EtaFull","Layers vs Eta 0th Full",28,-2.8,2.8,30,0,30.);
  zeroLayersvsEta[1] = dbe->book2D("Layers0EtaFast","Layers vs Eta 0th Fast",28,-2.8,2.8,30,0,30.);
  firstLayersvsEta[0] = dbe->book2D("Layers1EtaFull","Layers vs Eta 1st Full",28,-2.8,2.8,30,0,30.);
  firstLayersvsEta[1] = dbe->book2D("Layers1EtaFast","Layers vs Eta 1st Fast",28,-2.8,2.8,30,0,30.);
  secondLayersvsEta[0] = dbe->book2D("Layers2EtaFull","Layers vs Eta 2nd Full",28,-2.8,2.8,30,0,30.);
  secondLayersvsEta[1] = dbe->book2D("Layers2EtaFast","Layers vs Eta 2nd Fast",28,-2.8,2.8,30,0,30.);
  thirdLayersvsEta[0] = dbe->book2D("Layers3EtaFull","Layers vs Eta 3rd Full",28,-2.8,2.8,30,0,30.);
  thirdLayersvsEta[1] = dbe->book2D("Layers3EtaFast","Layers vs Eta 3rd Fast",28,-2.8,2.8,30,0,30.);
  fourthLayersvsEta[0] = dbe->book2D("Layers4EtaFull","Layers vs Eta 4th Full",28,-2.8,2.8,30,0,30.);
  fourthLayersvsEta[1] = dbe->book2D("Layers4EtaFast","Layers vs Eta 4th Fast",28,-2.8,2.8,30,0,30.);
  fifthLayersvsEta[0] = dbe->book2D("Layers5EtaFull","Layers vs Eta 5th Full",28,-2.8,2.8,30,0,30.);
  fifthLayersvsEta[1] = dbe->book2D("Layers5EtaFast","Layers vs Eta 5th Fast",28,-2.8,2.8,30,0,30.);

  thirdSeedvsP[0] = dbe->book2D("Seed3PFull","Seed vs P 3rd Full",100,0.,10.,10,0,10.);
  thirdSeedvsP[1] = dbe->book2D("Seed3PFast","Seed vs P 3rd Fast",100,0.,10.,10,0,10.);
  thirdSeedvsEta[0] = dbe->book2D("Seed3EtaFull","Seed vs Eta 3rd Full",28,-2.8,2.8,10,0,10.);
  thirdSeedvsEta[1] = dbe->book2D("Seed3EtaFast","Seed vs Eta 3rd Fast",28,-2.8,2.8,10,0,10.);
  fifthSeedvsP[0] = dbe->book2D("Seed5PFull","Seed vs P 5th Full",100,0.,10.,10,0,10.);
  fifthSeedvsP[1] = dbe->book2D("Seed5PFast","Seed vs P 5th Fast",100,0.,10.,10,0,10.);
  fifthSeedvsEta[0] = dbe->book2D("Seed5EtaFull","Seed vs Eta 5th Full",28,-2.8,2.8,10,0,10.);
  fifthSeedvsEta[1] = dbe->book2D("Seed5EtaFast","Seed vs Eta 5th Fast",28,-2.8,2.8,10,0,10.);


  zeroNum[0]  = dbe->book1D("Num0Full","Num 0th Full",10, 0., 10.);
  zeroNum[1]  = dbe->book1D("Num0Fast","Num 0th Fast",10, 0., 10.);
  firstNum[0]  = dbe->book1D("Num1Full","Num 1st Full",10, 0., 10.);
  firstNum[1]  = dbe->book1D("Num1Fast","Num 1st Fast",10, 0., 10.);
  secondNum[0] = dbe->book1D("Num2Full","Num 2nd Full",10, 0., 10.);
  secondNum[1] = dbe->book1D("Num2Fast","Num 2nd Fast",10, 0., 10.);
  thirdNum[0]  = dbe->book1D("Num3Full","Num 3rd Full",10, 0., 10.);
  thirdNum[1]  = dbe->book1D("Num3Fast","Num 3rd Fast",10, 0., 10.);
  fourthNum[0] = dbe->book1D("Num4Full","Num 4th Full",10, 0., 10.);
  fourthNum[1] = dbe->book1D("Num4Fast","Num 4th Fast",10, 0., 10.);
  fifthNum[0]  = dbe->book1D("Num5Full","Num 5th Full",10, 0., 10.);
  fifthNum[1]  = dbe->book1D("Num5Fast","Num 5th Fast",10, 0., 10.);

}

testTrackingIterations::~testTrackingIterations()
{

  std::cout << "\t\t ZERO \tFIRST \tSECOND \tTHIRD\t FOURTH\tFIFTH " << std::endl;
  std::cout << "\tFULL\t" <<  num0full << "\t"<< num1full << "\t" << num2full <<"\t" << num3full << "\t" << num4full <<  "\t" << num5full << std::endl;
  std::cout << "\tFAST\t" <<  num0fast <<"\t"<< num1fast << "\t" << num2fast << "\t" << num3fast << "\t" << num4fast << "\t" << num5fast << std::endl;

  dbe->save(outputFileName);

}

void testTrackingIterations::beginRun(edm::Run& run, edm::EventSetup const& es)
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

  num0fast = num1fast = num2fast = num3fast = num4fast= num5fast = 0;
  num0full = num1full = num2full = num3full = num4full= num5full = 0;

}

void
testTrackingIterations::produce(edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  iSetup.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();



  ++totalNEvt;

     std::cout << " >>>>>>>>> Analizying Event " << totalNEvt << "<<<<<<< " << std::endl; 

  if ( totalNEvt/1000*1000 == totalNEvt ) 
    std::cout << "Number of event analysed "
	      << totalNEvt << std::endl; 

  std::auto_ptr<edm::SimTrackContainer> nuclSimTracks(new edm::SimTrackContainer);


  // Fill sim events.
  //  std::cout << "Fill full event " << std::endl;
  edm::Handle<std::vector<SimTrack> > fullSimTracks;
  iEvent.getByLabel("g4SimHits",fullSimTracks);
  edm::Handle<std::vector<SimVertex> > fullSimVertices;
  iEvent.getByLabel("g4SimHits",fullSimVertices);
  mySimEvent[0]->fill( *fullSimTracks, *fullSimVertices );
  
  edm::Handle<std::vector<SimTrack> > fastSimTracks;
  iEvent.getByLabel("famosSimHits",fastSimTracks);
  edm::Handle<std::vector<SimVertex> > fastSimVertices;
  iEvent.getByLabel("famosSimHits",fastSimVertices);
  mySimEvent[1]->fill( *fastSimTracks, *fastSimVertices );

  // Get the GS RecHits
  //  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theGSRecHits;
  iEvent.getByLabel("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits", theGSRecHits);
  TrackerRecHit theFirstSeedingTrackerRecHit;


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

    //fill SimTrack Info first 
    for(unsigned int simi=0; simi < mySimEvent[ievt]->nTracks();++simi){
      const FSimTrack& theTrack = mySimEvent[1]->track(simi);
      double etaSim = theTrack.momentum().Eta();
      double pSim = std::sqrt(theTrack.momentum().Vect().Perp2());

      //      if(fabs(etaSim)<1.8) return;//skip the event
      SimTracksvsEtaP[ievt]->Fill(etaSim,pSim,1.);
    }

    edm::Handle<reco::TrackCollection> tkRef0;
    edm::Handle<reco::TrackCollection> tkRef1;
    edm::Handle<reco::TrackCollection> tkRef2;
    edm::Handle<reco::TrackCollection> tkRef3;
    edm::Handle<reco::TrackCollection> tkRef4;
    edm::Handle<reco::TrackCollection> tkRef5;
    iEvent.getByLabel(zeroTracks[ievt],tkRef0);    
    iEvent.getByLabel(firstTracks[ievt],tkRef1);    
    iEvent.getByLabel(secondTracks[ievt],tkRef2);    
    iEvent.getByLabel(thirdTracks[ievt],tkRef3);   
    iEvent.getByLabel(fourthTracks[ievt],tkRef4);   
    iEvent.getByLabel(fifthTracks[ievt],tkRef5);   
    std::vector<const reco::TrackCollection*> tkColl;
    tkColl.push_back(tkRef0.product());
    tkColl.push_back(tkRef1.product());
    tkColl.push_back(tkRef2.product());
    tkColl.push_back(tkRef3.product());
    tkColl.push_back(tkRef4.product());
    tkColl.push_back(tkRef5.product());
    
    //      if ( tkColl[0]+tkColl[1]+tkColl[2] != 1 ) continue;

    if(ievt ==0){
      num0full +=  tkColl[0]->size();       
      num1full +=  tkColl[1]->size();       
      num2full +=  tkColl[2]->size();
      num3full +=  tkColl[3]->size();
      num4full +=  tkColl[4]->size();
      num5full +=  tkColl[5]->size();
    } else if (ievt ==1){
      num0fast +=  tkColl[0]->size();
      num1fast +=  tkColl[1]->size();
      num2fast +=  tkColl[2]->size();
      num3fast +=  tkColl[3]->size();
      num4fast +=  tkColl[4]->size();
      num5fast +=  tkColl[5]->size();
    }
    
    if (DEBUG) {
      std::cout << " PArticle list: Pt = "  <<  pGen << " , eta = " << etaGen << std::endl;
      std::cout << "\t\t ZERO \tFIRST \tSECOND \tTHIRD\t FOURTH\tFIFTH " << std::endl;
      std::cout << "\t" << TheSample(ievt) << "\t" << tkColl[0]->size() <<"\t"
		<< tkColl[1]->size() <<"\t"<< tkColl[2]->size() <<"\t"
		<< tkColl[3]->size() <<"\t"<< tkColl[4]->size() <<"\t"
		<< tkColl[5]->size() << std::endl;
    }
    zeroNum[ievt]->Fill(tkColl[0]->size());
    firstNum[ievt]->Fill(tkColl[1]->size());
    secondNum[ievt]->Fill(tkColl[2]->size());
    thirdNum[ievt]->Fill(tkColl[3]->size());
    fourthNum[ievt]->Fill(tkColl[4]->size());
    fifthNum[ievt]->Fill(tkColl[5]->size());


    //    if ( tkColl[0]->size() == 1 ) { 
    reco::TrackCollection::const_iterator itk0 = tkColl[0]->begin();
    reco::TrackCollection::const_iterator itk0_e = tkColl[0]->end();
    for(;itk0!=itk0_e;++itk0){
      //      firstTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[0]->size());
      zeroTracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
      //      reco::TrackCollection::const_iterator itk = tkColl[0]->begin();
      zeroHitsvsEta[ievt]->Fill(etaGen,itk0->found(),1.);
      zeroHitsvsP[ievt]->Fill(pGen,itk0->found(),1.);
      zeroLayersvsEta[ievt]->Fill(etaGen,itk0->hitPattern().trackerLayersWithMeasurement(),1.);
      zeroLayersvsP[ievt]->Fill(pGen,itk0->hitPattern().trackerLayersWithMeasurement(),1.);
    }
    //    if ( tkColl[1]->size() == 1 ) { 
    reco::TrackCollection::const_iterator itk1 = tkColl[1]->begin();
    reco::TrackCollection::const_iterator itk1_e = tkColl[1]->end();
    for(;itk1!=itk1_e;++itk1){
      if (DEBUG) {
	std::cout << "FIRST ITER Track Pt = " << itk1->pt() << ", eta  = " << itk1->eta() << std::endl;
      }

      //      firstTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[0]->size());
      firstTracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
      //      reco::TrackCollection::const_iterator itk = tkColl[0]->begin();
      firstHitsvsEta[ievt]->Fill(etaGen,itk1->found(),1.);
      firstHitsvsP[ievt]->Fill(pGen,itk1->found(),1.);
      firstLayersvsEta[ievt]->Fill(etaGen,itk1->hitPattern().trackerLayersWithMeasurement(),1.);
      firstLayersvsP[ievt]->Fill(pGen,itk1->hitPattern().trackerLayersWithMeasurement(),1.);
    }
    //    if ( tkColl[2]->size() == 1 ) { 
    reco::TrackCollection::const_iterator itk2 = tkColl[2]->begin();
    reco::TrackCollection::const_iterator itk2_e = tkColl[2]->end();
    for(;itk2!=itk2_e;++itk2){
      secondTracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
      //      secondTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[1]->size());
      //      reco::TrackCollection::const_iterator itk = tkColl[1]->begin();
      secondHitsvsEta[ievt]->Fill(etaGen,itk2->found(),1.);    
      secondHitsvsP[ievt]->Fill(pGen,itk2->found(),1.);
      secondLayersvsEta[ievt]->Fill(etaGen,itk2->hitPattern().trackerLayersWithMeasurement(),1.);    
      secondLayersvsP[ievt]->Fill(pGen,itk2->hitPattern().trackerLayersWithMeasurement(),1.);
    }
    //    if ( tkColl[3]->size() == 1 ) { 
    reco::TrackCollection::const_iterator itk3 = tkColl[3]->begin();
    reco::TrackCollection::const_iterator itk3_e = tkColl[3]->end();
    for(;itk3!=itk3_e;++itk3){
      thirdTracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
      //      thirdTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[2]->size());
      //      reco::TrackCollection::const_iterator itk = tkColl[2]->begin();
      thirdHitsvsEta[ievt]->Fill(etaGen,itk3->found(),1.);
      thirdHitsvsP[ievt]->Fill(pGen,itk3->found(),1.);
      thirdLayersvsEta[ievt]->Fill(etaGen,itk3->hitPattern().trackerLayersWithMeasurement(),1.);
      thirdLayersvsP[ievt]->Fill(pGen,itk3->hitPattern().trackerLayersWithMeasurement(),1.);

      unsigned int firstSubDetId =-99;
      unsigned int secondSubDetId = -99;
      unsigned int firstID =-99;
      unsigned int firstLayerNumber=-99; 
      unsigned int secondLayerNumber =-99;
      
      //now find the seed type (i.e. where the seed hits are) 
      const TrajectorySeed* seed = itk3->seedRef().get();
      if (DEBUG) {
	std::cout << TheSample(ievt) << "SIM HITS IN THE SEED " << seed->nHits() << std::endl;
      }
      
      //      const GeomDet* theGeomDet;
      unsigned int theSubDetId=-99; 
      unsigned int theLayerNumber=-99;
      // unsigned int theRingNumber=-99;
      int hit=0;
      TrajectorySeed::range hitRange = seed->recHits();
      for( TrajectorySeed::const_iterator ihit = hitRange.first; ihit !=hitRange.second; ihit++){
	
	const DetId& theDetId = ihit->geographicalId();
	//	theGeomDet = theGeometry->idToDet(theDetId);
	theLayerNumber=tTopo->layer(theDetId);
	if(hit==0){
	  const SiTrackerGSMatchedRecHit2D * theFirstSeedingRecHit = (const SiTrackerGSMatchedRecHit2D*) (&(*(ihit)));
	  firstID = theFirstSeedingRecHit->simtrackId();
	  firstSubDetId =  theSubDetId;
	  firstLayerNumber = theLayerNumber;	
	  std::cout << "First Hit " << " Subdet " << firstSubDetId << ", Layer " << firstLayerNumber << std::endl;
	  if(ievt==1) std::cout	    << "Local Pos = " <<  ihit->localPosition()
		    << "Local error = " <<  ihit->localPositionError() << std::endl; 
	}
	if(hit ==1 ) { 
	  secondSubDetId =  theSubDetId;
	  secondLayerNumber = theLayerNumber;
	  std::cout << "Second Hit " << " Subdet " << secondSubDetId << ", Layer " << secondLayerNumber  << std::endl;
	  if(ievt==1) std::cout  << "Local Pos = " <<  ihit->localPosition() 
		    << "Local error = " << ihit->localPositionError() << std::endl; 
	}
	hit++;
      }
      if((firstSubDetId==1 && firstLayerNumber==1) && (secondSubDetId==1 && secondLayerNumber==2)) MySeedType3 = 1;//BPIX1+BPIX2
      if((firstSubDetId==1 && firstLayerNumber==2) && (secondSubDetId==1 && secondLayerNumber==3)) MySeedType3 = 2;//BPIX2+BPIX3
      if((firstSubDetId==1 && firstLayerNumber==1) && (secondSubDetId==1 && secondLayerNumber==1)) MySeedType3 = 3;//BPIX1+FPIX1
      if((firstSubDetId==2 && firstLayerNumber==1) && (secondSubDetId==2 && secondLayerNumber==2)) MySeedType3 = 4;//FPIX1+FPIX2
      if((firstSubDetId==1 && firstLayerNumber==1) && (secondSubDetId==6 && secondLayerNumber==2)) MySeedType3 = 5;//FPIX1+TEC2
      
     //  std::cout << "3rd Seed Type " << MySeedType3 << std::endl;

      thirdSeedvsEta[ievt]->Fill(etaGen,MySeedType3,1.);
      thirdSeedvsP[ievt]->Fill(pGen,MySeedType3,1.);

      if (DEBUG) {
	//get the Simtrack ID for this track for the FastSim 
	if(ievt==1) {
	  std::cout << "HERE Simtrack = " << firstID << "\tNUMBER of rechits" << theGSRecHits->size() << std::endl;
	  // Get all the rechits associated to this track
	  SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = theGSRecHits->get(firstID);
	  SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
	  SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
	  SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;
	  std::cout <<"counting: "<< theRecHitRangeIteratorEnd-theRecHitRangeIteratorBegin <<" hits to be considered "<< std::endl;
	  
	  int hitnum=0;
	  TrackerRecHit theCurrentRecHit;
	  for ( iterRecHit = theRecHitRangeIteratorBegin; 
		iterRecHit != theRecHitRangeIteratorEnd; 
		++iterRecHit) {
	    
	    theCurrentRecHit = TrackerRecHit(&(*iterRecHit),theGeometry,tTopo);
	    std::cout << hitnum << " Hit DetID = " <<   theCurrentRecHit.subDetId() << "\tLayer = " << theCurrentRecHit.layerNumber() << std::endl;	
	    hitnum++;
	  }
	}
      }  // End DEBUG

    }
    //    if ( tkColl[4]->size() == 1 ) { 
    reco::TrackCollection::const_iterator itk4 = tkColl[4]->begin();
    reco::TrackCollection::const_iterator itk4_e = tkColl[4]->end();
    for(;itk4!=itk4_e;++itk4){
      fourthTracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
      //      fourthTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[3]->size());
      //      reco::TrackCollection::const_iterator itk = tkColl[3]->begin();
      fourthHitsvsEta[ievt]->Fill(etaGen,itk4->found(),1.);
      fourthHitsvsP[ievt]->Fill(pGen,itk4->found(),1.);
      fourthLayersvsEta[ievt]->Fill(etaGen,itk4->hitPattern().trackerLayersWithMeasurement(),1.);
      fourthLayersvsP[ievt]->Fill(pGen,itk4->hitPattern().trackerLayersWithMeasurement(),1.);
    }
    //    if ( tkColl[5]->size() == 1 ) { 
    reco::TrackCollection::const_iterator itk5 = tkColl[5]->begin();
    reco::TrackCollection::const_iterator itk5_e = tkColl[5]->end();
    for(;itk5!=itk5_e;++itk5){
      fifthTracksvsEtaP[ievt]->Fill(etaGen,pGen,1.);
      //      fourthTracksvsEtaP[ievt]->Fill(etaGen,pGen,tkColl[3]->size());
      //      reco::TrackCollection::const_iterator itk = tkColl[3]->begin();
      fifthHitsvsEta[ievt]->Fill(etaGen,itk5->found(),1.);
      fifthHitsvsP[ievt]->Fill(pGen,itk5->found(),1.);
      fifthLayersvsEta[ievt]->Fill(etaGen,itk5->hitPattern().trackerLayersWithMeasurement(),1.);
      fifthLayersvsP[ievt]->Fill(pGen,itk5->hitPattern().trackerLayersWithMeasurement(),1.);

      if (DEBUG) {
	std::cout << "Evt " << totalNEvt << " PArticle list: Pt = "  <<  pGen << " , eta = " << etaGen << std::endl;
	std::cout << "Num of FULL SimTracks " <<  mySimEvent[0]->nTracks() << "\t Num of FAST SimTracks = " <<  mySimEvent[1]->nTracks()<< std::endl;

	std::cout << "\t\t ZERO \tFIRST \tSECOND \tTHIRD\t FOURTH\tFIFTH " << std::endl;
        std::cout << "\t" << TheSample(ievt) << "\t" << tkColl[0]->size() <<"\t"
		  << tkColl[1]->size() <<"\t"<< tkColl[2]->size() <<"\t"<< tkColl[3]->size()
		  <<"\t"<< tkColl[4]->size() <<"\t"<< tkColl[5]->size() << std::endl;
      }

      unsigned int firstSubDetId=-99;
      unsigned int secondSubDetId =-99;
      unsigned int firstID=-99;
      unsigned int firstLayerNumber=-99; 
      unsigned int secondLayerNumber=-99;
      
      //now find the seed type (i.e. where the seed hits are) 
      const TrajectorySeed* seed = itk5->seedRef().get();
      if (DEBUG) {
	std::cout << TheSample(ievt) << "SIM HITS IN THE SEED " << seed->nHits() << std::endl;
      }
      
      //      const GeomDet* theGeomDet;
      unsigned int theSubDetId=-99; 
      unsigned int theLayerNumber=-99;
      //      unsigned int theRingNumber=-99;
      int hit=0;
      TrajectorySeed::range hitRange = seed->recHits();
      for( TrajectorySeed::const_iterator ihit = hitRange.first; ihit !=hitRange.second; ihit++){
	
	const DetId& theDetId = ihit->geographicalId();
	//	theGeomDet = theGeometry->idToDet(theDetId);
	theLayerNumber=tTopo->layer(theDetId);

	if(hit==0){
	  const SiTrackerGSMatchedRecHit2D * theFirstSeedingRecHit = (const SiTrackerGSMatchedRecHit2D*) (&(*(ihit)));
	  firstID = theFirstSeedingRecHit->simtrackId();
	  firstSubDetId =  theSubDetId;
	  firstLayerNumber = theLayerNumber;	
	  std::cout << "First Hit " << " Subdet " << firstSubDetId << ", Layer " << firstLayerNumber << std::endl; 
	}
	if(hit ==1 ) { 
	  secondSubDetId =  theSubDetId;
	  secondLayerNumber = theLayerNumber;
	  std::cout << "Second Hit " << " Subdet " << secondSubDetId << ", Layer " << secondLayerNumber << std::endl; 
	}
	hit++;
      }

      if((firstSubDetId==5 && firstLayerNumber==1) && (secondSubDetId==5 && secondLayerNumber==2)) MySeedType5 = 1;
      if((firstSubDetId==5 && firstLayerNumber==1) && (secondSubDetId==6 && secondLayerNumber==1)) MySeedType5 = 2;
      if((firstSubDetId==6 && firstLayerNumber==1) && (secondSubDetId==6 && secondLayerNumber==2)) MySeedType5 = 3;
      if((firstSubDetId==6 && firstLayerNumber==2) && (secondSubDetId==6 && secondLayerNumber==3)) MySeedType5 = 4;
      if((firstSubDetId==6 && firstLayerNumber==3) && (secondSubDetId==6 && secondLayerNumber==4)) MySeedType5 = 5;
      if((firstSubDetId==6 && firstLayerNumber==4) && (secondSubDetId==6 && secondLayerNumber==5)) MySeedType5 = 6;
      if((firstSubDetId==6 && firstLayerNumber==5) && (secondSubDetId==6 && secondLayerNumber==6)) MySeedType5 = 7;
      if((firstSubDetId==6 && firstLayerNumber==6) && (secondSubDetId==6 && secondLayerNumber==7)) MySeedType5 = 8;
      
      std::cout << "5th Seed Type " << MySeedType5 << std::endl;

      fifthSeedvsEta[ievt]->Fill(etaGen,MySeedType5,1.);
      fifthSeedvsP[ievt]->Fill(pGen,MySeedType5,1.);

      if (DEBUG) {
	//get the Simtrack ID for this track for the FastSim 
	if(ievt==1) {
	  std::cout << "HERE Simtrack = " << firstID << "\tNUMBER of rechits" << theGSRecHits->size() << std::endl;
	  // Get all the rechits associated to this track
	  SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = theGSRecHits->get(firstID);
	  SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
	  SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
	  SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;
	  
	  std::cout <<"counting: "<< theRecHitRangeIteratorEnd-theRecHitRangeIteratorBegin <<" hits to be considered "<< std::endl;
	  
	  int hitnum=0;
	  TrackerRecHit theCurrentRecHit;
	  for ( iterRecHit = theRecHitRangeIteratorBegin; 
		iterRecHit != theRecHitRangeIteratorEnd; 
		++iterRecHit) {
	    
	    theCurrentRecHit = TrackerRecHit(&(*iterRecHit),theGeometry,tTopo);
	    std::cout << hitnum << " Hit DetID = " <<   theCurrentRecHit.subDetId() << "\tLayer = " << theCurrentRecHit.layerNumber() << std::endl;	
	    hitnum++;
	    
	  }
	  
	}
      }  // End DEBUG
      
    }
    
  }
  
}

//define this as a plug-in

DEFINE_FWK_MODULE(testTrackingIterations);
