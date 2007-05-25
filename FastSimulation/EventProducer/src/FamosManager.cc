// CMSSW Header
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Provenance/interface/EventID.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
//#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// HepMC headers
//#include "HepMC/GenEvent.h"

// FAMOS Header
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/EventProducer/interface/FamosManager.h"
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"
#include "FastSimulation/PileUpProducer/interface/PileUpSimulator.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"  
#include <iostream>
#include <memory>
#include <vector>

using namespace HepMC;

FamosManager::FamosManager(edm::ParameterSet const & p)
    : iEvent(0),
      //      myGenEvent(0),
      myPileUpSimulator(0),
      myCalorimetry(0),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_Tracking(p.getParameter<bool>("SimulateTracking")),
      m_Calorimetry(p.getParameter<bool>("SimulateCalorimetry")),
      m_PileUp(p.getParameter<bool>("SimulatePileUp")),
      m_pRunNumber(p.getUntrackedParameter<int>("RunNumber",1)),
      m_pVerbose(p.getUntrackedParameter<int>("Verbosity",1))
{

  // Initialize the random number generator service
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration")
      << "FamosManager requires the RandomGeneratorService\n"
         "which is not present in the configuration file.\n"
         "You must add the service in the configuration file\n"
         "or remove the module that requires it";
  }
  random = new RandomEngine(&(*rng));

  // Initialize the FSimEvent
  mySimEvent = 
    new FSimEvent(p.getParameter<edm::ParameterSet>("VertexGenerator"),
		  p.getParameter<edm::ParameterSet>("ParticleFilter"),
		  random);

  /// Initialize the TrajectoryManager
  myTrajectoryManager = 
    new TrajectoryManager(mySimEvent,
			  p.getParameter<edm::ParameterSet>("MaterialEffects"),
			  p.getParameter<edm::ParameterSet>("TrackerSimHits"),
			  p.getParameter<edm::ParameterSet>("ActivateDecays"),
			  random);

  // Initialize PileUp Producer (if requested)
  if ( m_PileUp ) {
    myPileUpSimulator = 
      new PileUpSimulator(mySimEvent,
			  p.getParameter<edm::ParameterSet>("PileUpSimulator"),
			  random);
  }

  // Initialize Calorimetry Fast Simulation (if requested)
  if ( m_Calorimetry) 
    myCalorimetry = 
      new CalorimetryManager(mySimEvent,
			     p.getParameter<edm::ParameterSet>("Calorimetry"),
			     random);

}

FamosManager::~FamosManager()
{ 
  if ( mySimEvent ) delete mySimEvent; 
  if ( myTrajectoryManager ) delete myTrajectoryManager; 
  if ( myPileUpSimulator ) delete myPileUpSimulator;
  if ( myCalorimetry) delete myCalorimetry;
}

void FamosManager::setupGeometryAndField(const edm::EventSetup & es)
{
  // Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  mySimEvent->initializePdt(&(*pdt));
  ParticleTable::instance(&(*pdt));

  // Geometry
  edm::ESHandle<DDCompactView> pDD;
  es.get<IdealGeometryRecord>().get(pDD);

  // Initialize the tracker reco geometry (always needed)
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;
  es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
  myTrajectoryManager->initializeRecoGeometry(&(*theGeomSearchTracker));

  // Initialize the full tracker geometry (only if tracking is requested)
  if ( m_Tracking ) {
    edm::ESHandle<TrackerGeometry> tracker;
    es.get<TrackerDigiGeometryRecord>().get(tracker);

    myTrajectoryManager->initializeTrackerGeometry(&(*tracker)); 

  }

  // magnetic field
  if (m_pUseMagneticField) {
    edm::ESHandle<MagneticField> pMF;
    es.get<IdealMagneticFieldRecord>().get(pMF);
    const GlobalPoint g(0.,0.,0.);
    std::cout << "B-field(T) at (0,0,0)(cm): " << pMF->inTesla(g) << std::endl;      
    MagneticFieldMap::instance( &(*pMF), myTrajectoryManager->theGeometry() ); 
 }    
  

  //  Initialize the calorimeter geometry
  if ( myCalorimetry ) {
    edm::ESHandle<CaloGeometry> pG;
    es.get<IdealGeometryRecord>().get(pG);   
    myCalorimetry->getCalorimeter()->setupGeometry(*pG);

    edm::ESHandle<CaloTopology> theCaloTopology;
    es.get<CaloTopologyRecord>().get(theCaloTopology);     
    myCalorimetry->getCalorimeter()->setupTopology(*theCaloTopology);
    myCalorimetry->getCalorimeter()->initialize();
  }

}


void 
FamosManager::reconstruct(const HepMC::GenEvent* evt,
			  const reco::CandidateCollection* particles) 
{

  //  myGenEvent = evt;

  if (evt != 0 || particles != 0) {
    iEvent++;
    edm::EventID id(m_pRunNumber,iEvent);


    // Fill the event from the original generated event
    if (evt ) 
      mySimEvent->fill(*evt,id);
    else
      mySimEvent->fill(*particles,id);


    //    mySimEvent->printMCTruth(*evt);
    /* 
    mySimEvent->print();
    std::cout << "----------------------------------------" << std::endl;
    */

    // Get the pileup events and add the particles to the main event
    if ( myPileUpSimulator ) { 
      myPileUpSimulator->produce();
      myPileUpSimulator->save();
    }

    /*
    mySimEvent->print();
    std::cout << "----------------------------------------" << std::endl;
    */

    // And propagate the particles through the detector
    myTrajectoryManager->reconstruct();
    /*
    mySimEvent->print();
    std::cout << "=========================================" 
	      << std::endl
	      << std::endl;
    */

    if ( myCalorimetry ) myCalorimetry->reconstruct();

  }

  std::cout << " saved : Event  " << iEvent 
	    << " of weight " << mySimEvent->weight()
	    << " with " << mySimEvent->nTracks() 
	    << " tracks and " << mySimEvent->nVertices()
	    << " vertices, generated by " 
	    << mySimEvent->nGenParts() << " particles " << std::endl;
  
}
