/** \class EcalDetailedTimeRecHitProducer
 *   produce ECAL detailed time Rechits
 *  \author Paolo Meridiani
 *
 **/

#include "RecoLocalCalo/EcalRecProducers/plugins/EcalDetailedTimeRecHitProducer.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <iostream>
#include <memory>

#include <vector>

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTimeDigi.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

EcalDetailedTimeRecHitProducer::EcalDetailedTimeRecHitProducer(const edm::ParameterSet& ps) : m_geometry(nullptr) {
  EBRecHitCollection_ = consumes<EBRecHitCollection>(ps.getParameter<edm::InputTag>("EBRecHitCollection"));
  EERecHitCollection_ = consumes<EERecHitCollection>(ps.getParameter<edm::InputTag>("EERecHitCollection"));

  ebTimeDigiCollection_ = consumes<EcalTimeDigiCollection>(ps.getParameter<edm::InputTag>("EBTimeDigiCollection"));
  eeTimeDigiCollection_ = consumes<EcalTimeDigiCollection>(ps.getParameter<edm::InputTag>("EETimeDigiCollection"));

  caloGeometry_ = esConsumes<CaloGeometry, CaloGeometryRecord>();

  EBDetailedTimeRecHitCollection_ = ps.getParameter<std::string>("EBDetailedTimeRecHitCollection");
  EEDetailedTimeRecHitCollection_ = ps.getParameter<std::string>("EEDetailedTimeRecHitCollection");

  correctForVertexZPosition_ = ps.getParameter<bool>("correctForVertexZPosition");
  useMCTruthVertex_ = ps.getParameter<bool>("useMCTruthVertex");
  if (correctForVertexZPosition_) {
    if (not useMCTruthVertex_) {
      recoVertex_ = consumes<reco::VertexCollection>(ps.getParameter<edm::InputTag>("recoVertex"));
    } else {
      simVertex_ = consumes<edm::SimVertexContainer>(ps.getParameter<edm::InputTag>("simVertex"));
    }
  }

  ebTimeLayer_ = ps.getParameter<int>("EBTimeLayer");
  eeTimeLayer_ = ps.getParameter<int>("EETimeLayer");

  produces<EBRecHitCollection>(EBDetailedTimeRecHitCollection_);
  produces<EERecHitCollection>(EEDetailedTimeRecHitCollection_);
}

EcalDetailedTimeRecHitProducer::~EcalDetailedTimeRecHitProducer() {}

void EcalDetailedTimeRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;
  using namespace reco;

  edm::ESHandle<CaloGeometry> hGeometry = es.getHandle(caloGeometry_);

  m_geometry = hGeometry.product();

  Handle<EBRecHitCollection> pEBRecHits;
  Handle<EERecHitCollection> pEERecHits;

  const EBRecHitCollection* EBRecHits = nullptr;
  const EERecHitCollection* EERecHits = nullptr;

  evt.getByToken(EBRecHitCollection_, pEBRecHits);
  if (pEBRecHits.isValid()) {
    EBRecHits = pEBRecHits.product();  // get a ptr to the product
#ifdef DEBUG
    LogDebug("EcalRecHitDebug") << "total # EB rechits to be re-calibrated: " << EBRecHits->size();
#endif
  }

  evt.getByToken(EERecHitCollection_, pEERecHits);
  if (pEERecHits.isValid()) {
    EERecHits = pEERecHits.product();  // get a ptr to the product
#ifdef DEBUG
    LogDebug("EcalRecHitDebug") << "total # EE uncalibrated rechits to be re-calibrated: " << EERecHits->size();
#endif
  }

  Handle<EcalTimeDigiCollection> pEBTimeDigis;
  Handle<EcalTimeDigiCollection> pEETimeDigis;

  const EcalTimeDigiCollection* ebTimeDigis = nullptr;
  const EcalTimeDigiCollection* eeTimeDigis = nullptr;

  evt.getByToken(ebTimeDigiCollection_, pEBTimeDigis);
  //evt.getByToken( digiProducer_, pEBTimeDigis);
  if (pEBTimeDigis.isValid()) {
    ebTimeDigis = pEBTimeDigis.product();  // get a ptr to the produc
    edm::LogInfo("EcalDetailedTimeRecHitInfo") << "total # ebTimeDigis: " << ebTimeDigis->size();
  }

  evt.getByToken(eeTimeDigiCollection_, pEETimeDigis);
  //evt.getByToken( digiProducer_, pEETimeDigis);
  if (pEETimeDigis.isValid()) {
    eeTimeDigis = pEETimeDigis.product();  // get a ptr to the product
    edm::LogInfo("EcalDetailedTimeRecHitInfo") << "total # eeTimeDigis: " << eeTimeDigis->size();
  }
  // collection of rechits to put in the event
  std::unique_ptr<EBRecHitCollection> EBDetailedTimeRecHits(new EBRecHitCollection);
  std::unique_ptr<EERecHitCollection> EEDetailedTimeRecHits(new EERecHitCollection);

  std::unique_ptr<GlobalPoint> vertex;

  if (correctForVertexZPosition_) {
    if (!useMCTruthVertex_) {
      //Get the first reco vertex
      // get primary vertices

      edm::Handle<VertexCollection> VertexHandle;
      evt.getByToken(recoVertex_, VertexHandle);

      if (VertexHandle.isValid()) {
        if (!(*VertexHandle).empty())  //at least 1 vertex
        {
          const reco::Vertex* myVertex = &(*VertexHandle)[0];
          vertex = std::make_unique<GlobalPoint>(myVertex->x(), myVertex->y(), myVertex->z());
        }
      }

    } else {
      edm::Handle<SimVertexContainer> VertexHandle;
      evt.getByToken(simVertex_, VertexHandle);

      if (VertexHandle.isValid()) {
        if (!(*VertexHandle).empty())  //at least 1 vertex
        {
          assert((*VertexHandle)[0].vertexId() == 0);
          const SimVertex* myVertex = &(*VertexHandle)[0];
          vertex = std::make_unique<GlobalPoint>(
              myVertex->position().x(), myVertex->position().y(), myVertex->position().z());
        }
      }
    }
  }

  if (EBRecHits && ebTimeDigis) {
    // loop over uncalibrated rechits to make calibrated ones
    for (EBRecHitCollection::const_iterator it = EBRecHits->begin(); it != EBRecHits->end(); ++it) {
      EcalRecHit aHit((*it));
      EcalTimeDigiCollection::const_iterator timeDigi = ebTimeDigis->find((*it).id());
      if (timeDigi != ebTimeDigis->end()) {
        if (timeDigi->sampleOfInterest() >= 0) {
          float myTime = (*timeDigi)[timeDigi->sampleOfInterest()];
          //Vertex corrected ToF
          if (vertex) {
            aHit.setTime(myTime + deltaTimeOfFlight(*vertex, (*it).id(), ebTimeLayer_));
          } else
            //Uncorrected ToF
            aHit.setTime(myTime);
        }
      }
      // leave standard time if no timeDigi is associated (e.g. noise recHits)
      EBDetailedTimeRecHits->push_back(aHit);
    }
  }

  if (EERecHits && eeTimeDigis) {
    // loop over uncalibrated rechits to make calibrated ones
    for (EERecHitCollection::const_iterator it = EERecHits->begin(); it != EERecHits->end(); ++it) {
      EcalRecHit aHit(*it);
      EcalTimeDigiCollection::const_iterator timeDigi = eeTimeDigis->find((*it).id());
      if (timeDigi != eeTimeDigis->end()) {
        if (timeDigi->sampleOfInterest() >= 0) {
          float myTime = (*timeDigi)[timeDigi->sampleOfInterest()];
          //Vertex corrected ToF
          if (vertex) {
            aHit.setTime(myTime + deltaTimeOfFlight(*vertex, (*it).id(), eeTimeLayer_));
          } else
            //Uncorrected ToF
            aHit.setTime(myTime);
        }
      }
      EEDetailedTimeRecHits->push_back(aHit);
    }
  }
  // put the collection of recunstructed hits in the event
  LogInfo("EcalDetailedTimeRecHitInfo") << "total # EB rechits: " << EBDetailedTimeRecHits->size();
  LogInfo("EcalDetailedTimeRecHitInfo") << "total # EE rechits: " << EEDetailedTimeRecHits->size();

  evt.put(std::move(EBDetailedTimeRecHits), EBDetailedTimeRecHitCollection_);
  evt.put(std::move(EEDetailedTimeRecHits), EEDetailedTimeRecHitCollection_);
}

double EcalDetailedTimeRecHitProducer::deltaTimeOfFlight(GlobalPoint& vertex, const DetId& detId, int layer) const {
  auto cellGeometry(m_geometry->getGeometry(detId));
  assert(nullptr != cellGeometry);
  GlobalPoint layerPos =
      cellGeometry->getPosition(double(layer) + 0.5);  //depth in mm in the middle of the layer position
  GlobalVector tofVector = layerPos - vertex;
  return (layerPos.mag() * cm - tofVector.mag() * cm) / (float)c_light;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalDetailedTimeRecHitProducer);
