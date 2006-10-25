// CMSSW Header
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/Common/interface/EventID.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// CLHEP headers
#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/JamesRandom.h"

// FAMOS Header
#include "FastSimulation/EventProducer/interface/FamosManager.h"
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"
#include "FastSimulation/PileUpProducer/interface/PUProducer.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"  
#include <iostream>
#include <memory>
#include <vector>

using namespace HepMC;
using namespace std;

FamosManager::FamosManager(edm::ParameterSet const & p)
    : iEvent(0),
      myGenEvent(0),
      mySimEvent(new FSimEvent(p.getParameter<edm::ParameterSet>("VertexGenerator"),
			       p.getParameter<edm::ParameterSet>("ParticleFilter"))),
      myTrajectoryManager(new TrajectoryManager
			      (mySimEvent,
			       p.getParameter<edm::ParameterSet>("MaterialEffects"),
			       p.getParameter<edm::ParameterSet>("TrackerSimHits"),
			       p.getParameter<bool>("ActivateDecays"))),
      myPileUpProducer(0),
      myCalorimetry(0),
      m_FamosSeed(p.getParameter<int>("FamosSeed")),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_Tracking(p.getParameter<bool>("SimulateTracking")),
      m_Calorimetry(p.getParameter<bool>("SimulateCalorimetry")),
      m_PileUp(p.getParameter<bool>("SimulatePileUp")),
      m_pRunNumber(p.getUntrackedParameter<int>("RunNumber",1)),
      m_pVerbose(p.getUntrackedParameter<int>("Verbosity",1))
{

  // Define the random generator engine for Famos
  HepRandom::setTheEngine(new HepJamesRandom());
  HepRandom::setTheSeeds(&m_FamosSeed,2);
  HepRandom::showEngineStatus(); 


  // Initialize PileUp Producer
  if ( m_PileUp ) 
    myPileUpProducer = new PUProducer(mySimEvent,
				      p.getParameter<edm::ParameterSet>("PUProducer"));

  // Initialize Calorimetry Fast Simulation
  if ( m_Calorimetry) 
    myCalorimetry = new CalorimetryManager(mySimEvent,
					   p.getParameter<edm::ParameterSet>("Calorimetry"));

}

FamosManager::~FamosManager()
{ 
  if ( mySimEvent ) delete mySimEvent; 
  if ( myTrajectoryManager ) delete myTrajectoryManager; 
  if ( myPileUpProducer ) delete myPileUpProducer;
  if ( myCalorimetry) delete myCalorimetry;
}

void FamosManager::setupGeometryAndField(const edm::EventSetup & es)
{
  // Particle data table (from Pythia)
  edm::ESHandle < DefaultConfig::ParticleDataTable > pdt;
  es.getData(pdt);
  mySimEvent->initializePdt(&(*pdt));
  ParticleTable::instance(&(*pdt));

  // Geometry
  edm::ESHandle<DDCompactView> pDD;
  es.get<IdealGeometryRecord>().get(pDD);

  // magnetic field
  if (m_pUseMagneticField) {
    edm::ESHandle<MagneticField> pMF;
    es.get<IdealMagneticFieldRecord>().get(pMF);
    const GlobalPoint g(0.,0.,0.);
    std::cout << "B-field(T) at (0,0,0)(cm): " << pMF->inTesla(g) << std::endl;      
    MagneticFieldMap::instance( &(*pMF) ); 
 }    
  
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
FamosManager::reconstruct(const HepMC::GenEvent* evt) {

  myGenEvent = evt;

  if (evt != 0) {
    iEvent++;
    edm::EventID id(m_pRunNumber,iEvent);


    // Fill the event from the original generated event
    mySimEvent->fill(*evt,id);
    //    mySimEvent->printMCTruth(*evt);

    // Get the pileup events and add the particles to the main event
    if ( myPileUpProducer ) myPileUpProducer->produce();
    //    mySimEvent->print();

    // And propagate the particles through the detector
    myTrajectoryManager->reconstruct();
    //    mySimEvent->print();

    if ( myCalorimetry ) myCalorimetry->reconstruct();

  }

  std::cout << " saved : Event  " << iEvent 
	    << " of weight " << mySimEvent->weight()
	    << " with " << mySimEvent->nTracks() 
	    << " tracks and " << mySimEvent->nVertices()
	    << " vertices, generated by " 
	    << mySimEvent->nGenParts() << " particles " << std::endl;
  
}
