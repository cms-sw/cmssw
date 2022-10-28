#ifndef Calibration_EcalCalibAlgos_Pi0FixedMassWindowCalibration_h
#define Calibration_EcalCalibAlgos_Pi0FixedMassWindowCalibration_h

#include <memory>
#include <string>
#include <iostream>

#include <vector>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// Framework
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

class Pi0FixedMassWindowCalibration : public edm::ESProducerLooper {
public:
  /// Constructor
  Pi0FixedMassWindowCalibration(const edm::ParameterSet& iConfig);

  /// Destructor
  ~Pi0FixedMassWindowCalibration() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// Dummy implementation (job done in duringLoop)
  virtual void produce(edm::Event&, const edm::EventSetup&){};

  /// Called at beginning of job
  void beginOfJob() override;

  /// Called at end of job
  void endOfJob() override;

  /// Called at beginning of loop
  void startingNewLoop(unsigned int iLoop) override;

  /// Called at end of loop
  Status endOfLoop(const edm::EventSetup&, unsigned int iLoop) override;

  /// Called at each event
  Status duringLoop(const edm::Event&, const edm::EventSetup&) override;

private:
  //  static const double PDGPi0Mass;

  int nevent;

  const unsigned int theMaxLoops;
  const std::string ecalHitsProducer_;
  const std::string barrelHits_;

  IslandClusterAlgo::VerbosityLevel verbosity;

  std::string barrelHitProducer_;
  std::string barrelHitCollection_;
  std::string barrelClusterCollection_;
  std::string clustershapecollectionEB_;
  std::string barrelClusterShapeAssociation_;

  PositionCalc posCalculator_;  // position calculation algorithm
  ClusterShapeAlgo shapeAlgo_;  // cluster shape algorithm
  IslandClusterAlgo* island_p;

  // Selection algorithm parameters
  double selePi0PtGammaOneMin_;
  double selePi0PtGammaTwoMin_;

  double selePi0DRBelt_;
  double selePi0DetaBelt_;

  double selePi0PtPi0Min_;

  double selePi0S4S9GammaOneMin_;
  double selePi0S4S9GammaTwoMin_;
  double selePi0S9S25GammaOneMin_;
  double selePi0S9S25GammaTwoMin_;

  double selePi0EtBeltIsoRatioMax_;

  double selePi0MinvMeanFixed_;
  double selePi0MinvSigmaFixed_;

  std::vector<DetId> barrelCells;

  // input calibration constants
  double oldCalibs_barl[85][360][2];
  double newCalibs_barl[85][360][2];
  double wxtals[85][360][2];
  double mwxtals[85][360][2];

  // steering parameters

  edm::ParameterSet theParameterSet;

  // map for all RecHits from ECal:
  std::map<DetId, EcalRecHit>* recHitsEB_map;

  const EcalRecHitCollection* ecalRecHitBarrelCollection;
  const EcalRecHitCollection* recalibEcalRecHitCollection;

  const edm::EDGetTokenT<EcalRecHitCollection> recHitToken_;
  const edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> intercalibConstantsToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;

  bool isfirstcall_;
};

#endif
