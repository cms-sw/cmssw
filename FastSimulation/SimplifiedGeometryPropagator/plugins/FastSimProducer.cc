// system include files
#include <memory>
#include <string>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESWatcher.h"

// data formats
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

// fastsim
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Geometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Decayer.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/LayerNavigator.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleFilter.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleManager.h"
#include "FastSimulation/Particle/interface/makeParticle.h"

// Hack for calorimetry
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "FastSimulation/ShowerDevelopment/interface/FastHFShowerLibrary.h"

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

//! The core class of the new SimplifiedGeometryPropagator.
/*!
    Coordinates the propagation of all particles, this means it does the following loop:
    1) Get particle from ParticleManager
    2) Call LayerNavigator to move particle to next intersection with layer
    3) Loop over all the interactions and add secondaries to the event
    4) Repeat steps 2), 3) until particle left the tracker, lost all its energy or is about to decay
    5) If particle is about to decay: do decay and add secondaries to the event
    6) Restart from 1) with the next particle
    7) If last particle was propagated add SimTracks, SimVertices, SimHits,... to the event
*/
class FastSimProducer : public edm::stream::EDProducer<> {
public:
  explicit FastSimProducer(const edm::ParameterSet&);
  ~FastSimProducer() override { ; }

private:
  void beginStream(edm::StreamID id) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;
  virtual FSimTrack createFSimTrack(fastsim::Particle* particle,
                                    fastsim::ParticleManager* particleManager,
                                    HepPDT::ParticleDataTable const& particleTable);

  edm::EDGetTokenT<edm::HepMCProduct> genParticlesToken_;  //!< Token to get the genParticles
  fastsim::Geometry geometry_;                             //!< The definition of the tracker according to python config
  fastsim::Geometry caloGeometry_;                         //!< Hack to interface "old" calo to "new" tracking
  double beamPipeRadius_;                                  //!< The radius of the beampipe
  double deltaRchargedMother_;              //!< Cut on deltaR for ClosestChargedDaughter algorithm (FastSim tracking)
  fastsim::ParticleFilter particleFilter_;  //!< Decides which particles have to be propagated
  std::unique_ptr<RandomEngineAndDistribution> _randomEngine;  //!< The random engine

  bool simulateCalorimetry;
  edm::ESWatcher<CaloGeometryRecord> watchCaloGeometry_;
  edm::ESWatcher<CaloTopologyRecord> watchCaloTopology_;
  std::unique_ptr<CalorimetryManager> myCalorimetry;  // unfortunately, default constructor cannot be called
  bool simulateMuons;
  
  bool fixLongLivedBug_;
  bool useFastSimsDecayer_;

  fastsim::Decayer decayer_;  //!< Handles decays of non-stable particles using pythia
  std::vector<std::unique_ptr<fastsim::InteractionModel> > interactionModels_;  //!< All defined interaction models
  std::map<std::string, fastsim::InteractionModel*> interactionModelMap_;  //!< Each interaction model has a unique name
  static const std::string MESSAGECATEGORY;  //!< Category of debugging messages ("FastSimulation")
};

const std::string FastSimProducer::MESSAGECATEGORY = "FastSimulation";

FastSimProducer::FastSimProducer(const edm::ParameterSet& iConfig)
    : genParticlesToken_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("src"))),
      geometry_(iConfig.getParameter<edm::ParameterSet>("trackerDefinition")),
      caloGeometry_(iConfig.getParameter<edm::ParameterSet>("caloDefinition")),
      beamPipeRadius_(iConfig.getParameter<double>("beamPipeRadius")),
      deltaRchargedMother_(iConfig.getParameter<double>("deltaRchargedMother")),
      particleFilter_(iConfig.getParameter<edm::ParameterSet>("particleFilter")),
      _randomEngine(nullptr),
      simulateCalorimetry(iConfig.getParameter<bool>("simulateCalorimetry")),
      simulateMuons(iConfig.getParameter<bool>("simulateMuons")),
      fixLongLivedBug_(iConfig.getParameter<bool>("fixLongLivedBug")),
      useFastSimsDecayer_(iConfig.getParameter<bool>("useFastSimsDecayer")) {
  //----------------
  // define interaction models
  //---------------

  const edm::ParameterSet& modelCfgs = iConfig.getParameter<edm::ParameterSet>("interactionModels");
  for (const std::string& modelName : modelCfgs.getParameterNames()) {
    const edm::ParameterSet& modelCfg = modelCfgs.getParameter<edm::ParameterSet>(modelName);
    std::string modelClassName(modelCfg.getParameter<std::string>("className"));
    // Use plugin-factory to create model
    std::unique_ptr<fastsim::InteractionModel> interactionModel(
        fastsim::InteractionModelFactory::get()->create(modelClassName, modelName, modelCfg));
    if (!interactionModel.get()) {
      throw cms::Exception("FastSimProducer") << "InteractionModel " << modelName << " could not be created";
    }
    // Add model to list
    interactionModels_.push_back(std::move(interactionModel));
    // and create the map
    interactionModelMap_[modelName] = interactionModels_.back().get();
  }

  //----------------
  // calorimetry
  //---------------

  if (simulateCalorimetry) {
    myCalorimetry.reset(new CalorimetryManager(nullptr,
                                               iConfig.getParameter<edm::ParameterSet>("Calorimetry"),
                                               iConfig.getParameter<edm::ParameterSet>("MaterialEffectsForMuonsInECAL"),
                                               iConfig.getParameter<edm::ParameterSet>("MaterialEffectsForMuonsInHCAL"),
                                               iConfig.getParameter<edm::ParameterSet>("GFlash")));
  }

  //----------------
  // register products
  //----------------

  // SimTracks and SimVertices
  produces<edm::SimTrackContainer>();
  produces<edm::SimVertexContainer>();
  // products of interaction models, i.e. simHits
  for (auto& interactionModel : interactionModels_) {
    interactionModel->registerProducts(*this);
  }
  produces<edm::PCaloHitContainer>("EcalHitsEB");
  produces<edm::PCaloHitContainer>("EcalHitsEE");
  produces<edm::PCaloHitContainer>("EcalHitsES");
  produces<edm::PCaloHitContainer>("HcalHits");
  produces<edm::SimTrackContainer>("MuonSimTracks");
}

void FastSimProducer::beginStream(const edm::StreamID id) {
  _randomEngine = std::make_unique<RandomEngineAndDistribution>(id);
}

void FastSimProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogDebug(MESSAGECATEGORY) << "   produce";

  geometry_.update(iSetup, interactionModelMap_);
  caloGeometry_.update(iSetup, interactionModelMap_);

  // Define containers for SimTracks, SimVertices
  std::unique_ptr<edm::SimTrackContainer> simTracks_(new edm::SimTrackContainer);
  std::unique_ptr<edm::SimVertexContainer> simVertices_(new edm::SimVertexContainer);

  // Get the particle data table (in case lifetime or charge of GenParticles not set)
  edm::ESHandle<HepPDT::ParticleDataTable> pdt;
  iSetup.getData(pdt);

  // Get the GenParticle collection
  edm::Handle<edm::HepMCProduct> genParticles;
  iEvent.getByToken(genParticlesToken_, genParticles);

  // Load the ParticleManager which returns the particles that have to be propagated
  // Creates a fastsim::Particle out of a GenParticle/secondary
  fastsim::ParticleManager particleManager(*genParticles->GetEvent(),
                                           *pdt,
                                           beamPipeRadius_,
                                           deltaRchargedMother_,
                                           particleFilter_,
                                           *simTracks_,
                                           *simVertices_,
                                           fixLongLivedBug_,
                                           useFastSimsDecayer_);

  //  Initialize the calorimeter geometry
  if (simulateCalorimetry) {
    if (watchCaloGeometry_.check(iSetup) || watchCaloTopology_.check(iSetup)) {
      edm::ESHandle<CaloGeometry> pG;
      iSetup.get<CaloGeometryRecord>().get(pG);
      myCalorimetry->getCalorimeter()->setupGeometry(*pG);

      edm::ESHandle<CaloTopology> theCaloTopology;
      iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
      myCalorimetry->getCalorimeter()->setupTopology(*theCaloTopology);
      myCalorimetry->getCalorimeter()->initialize(geometry_.getMagneticFieldZ(math::XYZTLorentzVector(0., 0., 0., 0.)));

      myCalorimetry->getHFShowerLibrary()->initHFShowerLibrary(iSetup);
    }

    // Important: this also cleans the calorimetry information from the last event
    myCalorimetry->initialize(_randomEngine.get());
  }

  // The vector of SimTracks needed for the CalorimetryManager
  std::vector<FSimTrack> myFSimTracks;

  LogDebug(MESSAGECATEGORY) << "################################"
                            << "\n###############################";

  // loop over particles
  for (std::unique_ptr<fastsim::Particle> particle = particleManager.nextParticle(*_randomEngine); particle != nullptr;
       particle = particleManager.nextParticle(*_randomEngine)) {
    LogDebug(MESSAGECATEGORY) << "\n   moving NEXT particle: " << *particle;

    // -----------------------------
    // This condition is necessary because of hack for calorimetry
    // -> The CalorimetryManager should also be implemented based on this new FastSim classes (Particle.h) in a future project.
    // A second loop (below) loops over all parts of the calorimetry in order to create a track of the old FastSim class FSimTrack.
    // The condition below (R<128, z<302) makes sure that the particle geometrically is outside the tracker boundaries
    // -----------------------------

    if (particle->position().Perp2() < 128. * 128. && std::abs(particle->position().Z()) < 302.) {
      // move the particle through the layers
      fastsim::LayerNavigator layerNavigator(geometry_);
      const fastsim::SimplifiedGeometry* layer = nullptr;

      // moveParticleToNextLayer(..) returns 0 in case that particle decays
      // in this case particle is propagated up to its decay vertex
      while (layerNavigator.moveParticleToNextLayer(*particle, layer)) {
        LogDebug(MESSAGECATEGORY) << "   moved to next layer: " << *layer;
        LogDebug(MESSAGECATEGORY) << "   new state: " << *particle;

        // Hack to interface "old" calo to "new" tracking
        // Particle reached calorimetry so stop further propagation
        if (layer->getCaloType() == fastsim::SimplifiedGeometry::TRACKERBOUNDARY) {
          layer = nullptr;
          // particle no longer is on a layer
          particle->resetOnLayer();
          break;
        }

        // break after 25 ns: only happens for particles stuck in loops
        if (particle->position().T() > 25) {
          layer = nullptr;
          // particle no longer is on a layer
          particle->resetOnLayer();
          break;
        }

        // perform interaction between layer and particle
        // do only if there is actual material
        if (layer->getThickness(particle->position(), particle->momentum()) > 1E-10) {
          int nSecondaries = 0;
          // loop on interaction models
          for (fastsim::InteractionModel* interactionModel : layer->getInteractionModels()) {
            LogDebug(MESSAGECATEGORY) << "   interact with " << *interactionModel;
            std::vector<std::unique_ptr<fastsim::Particle> > secondaries;
            interactionModel->interact(*particle, *layer, secondaries, *_randomEngine);
            nSecondaries += secondaries.size();
            particleManager.addSecondaries(particle->position(), particle->simTrackIndex(), secondaries, layer);
          }

          // kinematic cuts: particle might e.g. lost all its energy
          if (!particleFilter_.acceptsEn(*particle)) {
            // Add endvertex if particle did not create any secondaries
            if (nSecondaries == 0)
              particleManager.addEndVertex(particle.get());
            layer = nullptr;
            break;
          }
        }

        LogDebug(MESSAGECATEGORY) << "--------------------------------"
                                  << "\n-------------------------------";
      }

      // do decays
      if (!particle->isStable() && particle->remainingProperLifeTimeC() < 1E-10) {
        LogDebug(MESSAGECATEGORY) << "Decaying particle...";
        std::vector<std::unique_ptr<fastsim::Particle> > secondaries;
        if (useFastSimsDecayer_) decayer_.decay(*particle, secondaries, _randomEngine->theEngine());
        LogDebug(MESSAGECATEGORY) << "   decay has " << secondaries.size() << " products";
        particleManager.addSecondaries(particle->position(), particle->simTrackIndex(), secondaries);
        continue;
      }

      LogDebug(MESSAGECATEGORY) << "################################"
                                << "\n###############################";
    }

    // -----------------------------
    // Hack to interface "old" calorimetry with "new" propagation in tracker
    // The CalorimetryManager has to know which particle could in principle hit which parts of the calorimeter
    // I think it's a bit strange to propagate the particle even further (and even decay it) if it already hits
    // some part of the calorimetry but this is how the code works...
    // -----------------------------

    if (particle->position().Perp2() >= 128. * 128. || std::abs(particle->position().Z()) >= 302.) {
      LogDebug(MESSAGECATEGORY) << "\n   moving particle to calorimetry: " << *particle;

      // create FSimTrack (this is the object the old propagation uses)
      myFSimTracks.push_back(createFSimTrack(particle.get(), &particleManager, *pdt));
      // particle was decayed
      if (!particle->isStable() && particle->remainingProperLifeTimeC() < 1E-10) {
        continue;
      }

      LogDebug(MESSAGECATEGORY) << "################################"
                                << "\n###############################";
    }

    // -----------------------------
    // End Hack
    // -----------------------------

    LogDebug(MESSAGECATEGORY) << "################################"
                              << "\n###############################";
  }

  // store simTracks and simVertices
  iEvent.put(std::move(simTracks_));
  iEvent.put(std::move(simVertices_));
  // store products of interaction models, i.e. simHits
  for (auto& interactionModel : interactionModels_) {
    interactionModel->storeProducts(iEvent);
  }

  // -----------------------------
  // Calorimetry Manager
  // -----------------------------
  if (simulateCalorimetry) {
    for (auto myFSimTrack : myFSimTracks) {
      myCalorimetry->reconstructTrack(myFSimTrack, _randomEngine.get());
    }
  }

  // -----------------------------
  // Store Hits
  // -----------------------------
  std::unique_ptr<edm::PCaloHitContainer> p4(new edm::PCaloHitContainer);
  std::unique_ptr<edm::PCaloHitContainer> p5(new edm::PCaloHitContainer);
  std::unique_ptr<edm::PCaloHitContainer> p6(new edm::PCaloHitContainer);
  std::unique_ptr<edm::PCaloHitContainer> p7(new edm::PCaloHitContainer);

  std::unique_ptr<edm::SimTrackContainer> m1(new edm::SimTrackContainer);

  if (simulateCalorimetry) {
    myCalorimetry->loadFromEcalBarrel(*p4);
    myCalorimetry->loadFromEcalEndcap(*p5);
    myCalorimetry->loadFromPreshower(*p6);
    myCalorimetry->loadFromHcal(*p7);
    if (simulateMuons) {
      myCalorimetry->harvestMuonSimTracks(*m1);
    }
  }
  iEvent.put(std::move(p4), "EcalHitsEB");
  iEvent.put(std::move(p5), "EcalHitsEE");
  iEvent.put(std::move(p6), "EcalHitsES");
  iEvent.put(std::move(p7), "HcalHits");
  iEvent.put(std::move(m1), "MuonSimTracks");
}

void FastSimProducer::endStream() { _randomEngine.reset(); }

FSimTrack FastSimProducer::createFSimTrack(fastsim::Particle* particle,
                                           fastsim::ParticleManager* particleManager,
                                           HepPDT::ParticleDataTable const& particleTable) {
  FSimTrack myFSimTrack(particle->pdgId(),
                        particleManager->getSimTrack(particle->simTrackIndex()).momentum(),
                        particle->simVertexIndex(),
                        particle->genParticleIndex(),
                        particle->simTrackIndex(),
                        particle->charge(),
                        particle->position(),
                        particle->momentum(),
                        particleManager->getSimVertex(particle->simVertexIndex()));

  // move the particle through the caloLayers
  fastsim::LayerNavigator caloLayerNavigator(caloGeometry_);
  const fastsim::SimplifiedGeometry* caloLayer = nullptr;

  // moveParticleToNextLayer(..) returns 0 in case that particle decays
  // in this case particle is propagated up to its decay vertex
  while (caloLayerNavigator.moveParticleToNextLayer(*particle, caloLayer)) {
    LogDebug(MESSAGECATEGORY) << "   moved to next caloLayer: " << *caloLayer;
    LogDebug(MESSAGECATEGORY) << "   new state: " << *particle;

    // break after 25 ns: only happens for particles stuck in loops
    if (particle->position().T() > 50) {
      caloLayer = nullptr;
      break;
    }

    //////////
    // Define ParticlePropagators (RawParticle) needed for CalorimetryManager and save them
    //////////

    RawParticle PP = makeParticle(&particleTable, particle->pdgId(), particle->momentum(), particle->position());

    // no material
    if (caloLayer->getThickness(particle->position(), particle->momentum()) < 1E-10) {
      // unfortunately needed for CalorimetryManager
      if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::ECAL) {
        if (!myFSimTrack.onEcal()) {
          myFSimTrack.setEcal(PP, 0);
        }
      } else if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::HCAL) {
        if (!myFSimTrack.onHcal()) {
          myFSimTrack.setHcal(PP, 0);
        }
      } else if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::VFCAL) {
        if (!myFSimTrack.onVFcal()) {
          myFSimTrack.setVFcal(PP, 0);
        }
      }

      // not necessary to continue propagation
      if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::VFCAL) {
        myFSimTrack.setGlobal();
        caloLayer = nullptr;
        break;
      }

      continue;
    }

    // Stupid variable used by the old propagator
    // For details check BaseParticlePropagator.h
    int success = 0;
    if (caloLayer->isForward()) {
      success = 2;
      // particle moves inwards
      if (particle->position().Z() * particle->momentum().Z() < 0) {
        success *= -1;
      }
    } else {
      success = 1;
      // particle moves inwards
      if (particle->momentum().X() * particle->position().X() + particle->momentum().Y() * particle->position().Y() <
          0) {
        success *= -1;
      }
    }

    // Save the hit
    if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::PRESHOWER1) {
      if (!myFSimTrack.onLayer1()) {
        myFSimTrack.setLayer1(PP, success);
      }
    }

    if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::PRESHOWER2) {
      if (!myFSimTrack.onLayer2()) {
        myFSimTrack.setLayer2(PP, success);
      }
    }

    if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::ECAL) {
      if (!myFSimTrack.onEcal()) {
        myFSimTrack.setEcal(PP, success);
      }
    }

    if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::HCAL) {
      if (!myFSimTrack.onHcal()) {
        myFSimTrack.setHcal(PP, success);
      }
    }

    if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::VFCAL) {
      if (!myFSimTrack.onVFcal()) {
        myFSimTrack.setVFcal(PP, success);
      }
    }

    // Particle reached end of detector
    if (caloLayer->getCaloType() == fastsim::SimplifiedGeometry::VFCAL) {
      myFSimTrack.setGlobal();
      caloLayer = nullptr;
      break;
    }

    LogDebug(MESSAGECATEGORY) << "--------------------------------"
                              << "\n-------------------------------";
  }

  // do decays
  // don't have to worry about daughters if particle already within the calorimetry
  // since they will be rejected by the vertex cut of the ParticleFilter
  if (!particle->isStable() && particle->remainingProperLifeTimeC() < 1E-10) {
    LogDebug(MESSAGECATEGORY) << "Decaying particle...";
    std::vector<std::unique_ptr<fastsim::Particle> > secondaries;
    if (useFastSimsDecayer_) decayer_.decay(*particle, secondaries, _randomEngine->theEngine());
    LogDebug(MESSAGECATEGORY) << "   decay has " << secondaries.size() << " products";
    particleManager->addSecondaries(particle->position(), particle->simTrackIndex(), secondaries);
  }

  return myFSimTrack;
}

DEFINE_FWK_MODULE(FastSimProducer);
