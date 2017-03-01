// -*- C++ -*-
//
// Package:    testCaloCaloGeometryTools
// Class:      testCaloCaloGeometryTools
// 
/**\class testCaloCaloGeometryTools testEcalHitMaker.cc test/testEcalHitMaker/src/testEcalHitMaker.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FastSimulation/ShowerDevelopment/interface/EMECALShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/EMShower.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/Utilities/interface/GammaFunctionGenerator.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <iomanip>

//
// class decleration
//

typedef math::XYZVector XYZVector;
typedef math::XYZVector XYZPoint;



class testEcalHitMaker : public edm::stream::EDAnalyzer <> {
public:
  explicit testEcalHitMaker( const edm::ParameterSet& );
  ~testEcalHitMaker();
  virtual void beginRun(edm::Run const&, edm::EventSetup const& );

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  void testBorderCrossing();
  int pass_;

  CaloGeometryHelper * myGeometry;

  GammaFunctionGenerator* aGammaGenerator;
  FSimEvent * mySimEvent;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
testEcalHitMaker::testEcalHitMaker( const edm::ParameterSet& iConfig )
{
  aGammaGenerator = new GammaFunctionGenerator();

  mySimEvent = new FSimEvent(iConfig.getParameter<edm::ParameterSet>( "TestParticleFilter" ));
  
  myGeometry = new CaloGeometryHelper(iConfig.getParameter<edm::ParameterSet>("Calorimetry"));
}

void testEcalHitMaker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup){
   edm::ESHandle<CaloTopology> theCaloTopology;
   iSetup.get<CaloTopologyRecord>().get(theCaloTopology);     

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);     


   // Setup the tools
   double bField000 = 4.;
   myGeometry->setupGeometry(*pG);
   myGeometry->setupTopology(*theCaloTopology);
   myGeometry->initialize(bField000);

    // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  iSetup.getData(pdt);
  mySimEvent->initializePdt(&(*pdt));
  std::cout << " done with beginRun " << std::endl;
}


testEcalHitMaker::~testEcalHitMaker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to produce the data  ------------
void
testEcalHitMaker::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   RandomEngineAndDistribution random(iEvent.streamID());

   math::XYZTLorentzVectorD theMomentum(10.,0.,5.,sqrt(125));

   // no need actually define it at the ECAL entrance: the fill of FSimEvent will do the 
   // propagation
   math::XYZVectorD thePositionatEcalEntrance(129.,0.,60);
   
   std::vector<SimTrack> mySimTracks;
   SimTrack myTrack(11,theMomentum,0,-1,thePositionatEcalEntrance,theMomentum);
   mySimTracks.push_back(myTrack);
   std::vector<SimVertex> mySimVertices;
   mySimVertices.push_back(SimVertex(thePositionatEcalEntrance,0.));

   mySimEvent->fill(mySimTracks,mySimVertices);
 
//   RawParticle myPart(11,theMomentum);
//   myPart.setVertex(thePositionatEcalEntrance.X(),
//		    thePositionatEcalEntrance.Y(),
//		    thePositionatEcalEntrance.Z(),
//		    0.);
//   FSimTrack mySimTrack(&myPart,-1,-1,11,mySimEvent);
//   mySimTrack.setTkPosition(thePositionatEcalEntrance);
//   mySimTrack.setTkMomentum(theMomentum);
//   mySimTrack.setEcal(myPart,1);
//   // put dummy quantities
//   mySimTrack.setLayer1(RawParticle(),0);
//   mySimTrack.setLayer2(RawParticle(),0);
   
   
   FSimTrack  & mySimTrack (mySimEvent->track(0));
   std::cout << mySimTrack << std::endl;
   std::cout << " done " << std::endl;
   RawParticle myPart = mySimTrack.ecalEntrance(); 
   std::vector<const RawParticle*> thePart;
   thePart.push_back(&myPart);
   // no preshower
   XYZPoint ecalentrance = myPart.vertex().Vect();
 
   std::cout << " on ECAL/HCAL etc " << mySimTrack.onEcal() << " ";
   std::cout << mySimTrack.onHcal() << " "  ;
   std::cout << mySimTrack.onLayer1() << " "  ;
   std::cout << mySimTrack.onLayer2() << " " << std::endl;
   
   // ask me if you want details - makes the simulation faster
   std::vector<double> coreParams;
   coreParams.push_back(100);
   coreParams.push_back(0.1);
   std::vector<double> tailParams;
   tailParams.push_back(1);
   tailParams.push_back(0.1);
   tailParams.push_back(100);
   tailParams.push_back(1);

   // define the calorimeter properties 
   EMECALShowerParametrization 
     showerparam(myGeometry->ecalProperties(mySimTrack.onEcal()), 
		 myGeometry->hcalProperties(mySimTrack.onHcal()), 
		 myGeometry->layer1Properties(mySimTrack.onLayer1()), 
		 myGeometry->layer2Properties(mySimTrack.onLayer2()),
		 coreParams,
		 tailParams);

   //define the shower parameters 
   EMShower theShower(&random,aGammaGenerator,&showerparam,&thePart);
   
   // you might want to replace this with something elese 
   DetId pivot(myGeometry->getClosestCell(ecalentrance, true, mySimTrack.onEcal()==1));
   
   // define the 7x7 grid
   EcalHitMaker myGrid(myGeometry,ecalentrance,pivot,mySimTrack.onEcal(),7,0,&random);
   myGrid.setCrackPadSurvivalProbability(0.9); // current parameters  in the Fast Sim
   myGrid.setRadiusFactor(1.096); // current parameters 


   // define the track parameters 
   myGrid.setTrackParameters(myPart.Vect().Unit(),0.,mySimTrack);
   
   HcalHitMaker myHcalHitMaker(myGrid,(unsigned)0); 
   
   theShower.setGrid(&myGrid);
   theShower.setHcal(&myHcalHitMaker);
   theShower.compute();
   
   // print the result
   std::map<CaloHitID,float>::const_iterator mapitr;
   std::map<CaloHitID,float>::const_iterator endmapitr=myGrid.getHits().end();
   for(mapitr=myGrid.getHits().begin();mapitr!=endmapitr;++mapitr)
     {
       if(mapitr->second!=0)
	 std::cout << "DetId " <<EBDetId(mapitr->first.unitID()) << " " <<  std::setw(8) << std::setprecision(4)  << mapitr->second << std::endl;
     }
   
}



//define this as a plug-in
//DEFINE_SEAL_MODULE();
//DEFINE_ANOTHER_FWK_MODULE(testEcalHitMaker);
DEFINE_FWK_MODULE(testEcalHitMaker);
