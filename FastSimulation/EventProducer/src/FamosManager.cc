// CMSSW Header
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
//#include "Geometry/CaloTopology/interface/CaloTopology.h"

// HepMC headers
//#include "HepMC/GenEvent.h"

// FAMOS Header
#include "FastSimulation/EventProducer/interface/FamosManager.h"
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"  
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "FastSimulation/ShowerDevelopment/interface/FastHFShowerLibrary.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace HepMC;

FamosManager::FamosManager(edm::ParameterSet const & p)
    : iEvent(0),
      myCalorimetry(0),
      m_pUseMagneticField(p.getParameter<bool>("UseMagneticField")),
      m_Tracking(p.getParameter<bool>("SimulateTracking")),
      m_Calorimetry(p.getParameter<bool>("SimulateCalorimetry")),
      m_Alignment(p.getParameter<bool>("ApplyAlignment")),
      m_pRunNumber(p.getUntrackedParameter<int>("RunNumber",1)),
      m_pVerbose(p.getUntrackedParameter<int>("Verbosity",1))
{
  // Initialize the FSimEvent
  mySimEvent = 
    new FSimEvent(p.getParameter<edm::ParameterSet>("ParticleFilter"));

  /// Initialize the TrajectoryManager
  myTrajectoryManager = 
    new TrajectoryManager(mySimEvent,
			  p.getParameter<edm::ParameterSet>("MaterialEffects"),
			  p.getParameter<edm::ParameterSet>("TrackerSimHits"),
			  p.getParameter<edm::ParameterSet>("ActivateDecays"));

  // Initialize Calorimetry Fast Simulation (if requested)
  if ( m_Calorimetry) 
    myCalorimetry = 
      new CalorimetryManager(mySimEvent,
			     p.getParameter<edm::ParameterSet>("Calorimetry"),			     
			     p.getParameter<edm::ParameterSet>("MaterialEffectsForMuonsInECAL"),
			     p.getParameter<edm::ParameterSet>("MaterialEffectsForMuonsInHCAL"),
                             p.getParameter<edm::ParameterSet>("GFlash"));
}

FamosManager::~FamosManager()
{ 
  if ( mySimEvent ) delete mySimEvent; 
  if ( myTrajectoryManager ) delete myTrajectoryManager; 
  if ( myCalorimetry) delete myCalorimetry;
}

void 
FamosManager::setupGeometryAndField(edm::Run const& run, const edm::EventSetup & es)
{
  // Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  mySimEvent->initializePdt(&(*pdt));
  
  // Initialize the full (misaligned) tracker geometry 
  // (only if tracking is requested)
  std::string misAligned = m_Alignment ? "MisAligned" : "";
  // 1) By default, the aligned geometry is chosen (m_Alignment = false)
  // 2) By default, the misaligned geometry is aligned
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(misAligned,tracker);
  if (m_Tracking)  myTrajectoryManager->initializeTrackerGeometry(&(*tracker)); 

  // Initialize the tracker misaligned reco geometry (always needed)
  // By default, the misaligned geometry is aligned
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;
  es.get<TrackerRecoGeometryRecord>().get(misAligned, theGeomSearchTracker );

  // Initialize the misaligned tracker interaction geometry 
  edm::ESHandle<TrackerInteractionGeometry>  theTrackerInteractionGeometry;
  es.get<TrackerInteractionGeometryRecord>().get(misAligned, theTrackerInteractionGeometry );

  // Initialize the magnetic field
  double bField000 = 0.;
  if (m_pUseMagneticField) {
    edm::ESHandle<MagneticFieldMap> theMagneticFieldMap;
    es.get<MagneticFieldMapRecord>().get(misAligned, theMagneticFieldMap);
    const GlobalPoint g(0.,0.,0.);
    bField000 = theMagneticFieldMap->inTeslaZ(g);
    myTrajectoryManager->initializeRecoGeometry(&(*theGeomSearchTracker),
						&(*theTrackerInteractionGeometry),
						&(*theMagneticFieldMap));
  } else { 
    myTrajectoryManager->initializeRecoGeometry(&(*theGeomSearchTracker),
						&(*theTrackerInteractionGeometry),
						0);
    bField000 = 4.0;
  }
  // The following should be on LogInfo
  //std::cout << "B-field(T) at (0,0,0)(cm): " << bField000 << std::endl;      
    
  //  Initialize the calorimeter geometry
  if ( myCalorimetry ) {
    edm::ESHandle<CaloGeometry> pG;
    es.get<CaloGeometryRecord>().get(pG);   
    myCalorimetry->getCalorimeter()->setupGeometry(*pG);

    edm::ESHandle<CaloTopology> theCaloTopology;
    es.get<CaloTopologyRecord>().get(theCaloTopology);     
    myCalorimetry->getCalorimeter()->setupTopology(*theCaloTopology);
    myCalorimetry->getCalorimeter()->initialize(bField000);

    myCalorimetry->getHFShowerLibrary()->initHFShowerLibrary(es);
  }

  m_pRunNumber = run.run();

}


void 
FamosManager::reconstruct(const HepMC::GenEvent* evt,
			  const TrackerTopology *tTopo,
                          RandomEngineAndDistribution const* random)
{
  //  myGenEvent = evt;
  
  iEvent++;
  edm::EventID id(m_pRunNumber,1U,iEvent);


  // Fill the event from the original generated event
  mySimEvent->fill(*evt,id);
        
  // And propagate the particles through the detector
  myTrajectoryManager->reconstruct(tTopo, random);

  if ( myCalorimetry ) myCalorimetry->reconstruct(random);

  edm::LogInfo("FamosManager")  << " saved : Event  " << iEvent
				<< " of weight " << mySimEvent->weight()
				<< " with " << mySimEvent->nTracks()
				<< " tracks and " << mySimEvent->nVertices()
				<< " vertices, generated by "
				<< mySimEvent->nGenParts() << " particles " << std::endl;
}
