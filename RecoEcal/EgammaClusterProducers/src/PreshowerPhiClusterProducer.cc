#include <vector>
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <fstream>

#include "RecoEcal/EgammaClusterProducers/interface/PreshowerPhiClusterProducer.h"

using namespace edm;
using namespace std;

PreshowerPhiClusterProducer::PreshowerPhiClusterProducer(const edm::ParameterSet& ps) {
  // use configuration file to setup input/output collection names
  preshHitToken_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("preshRecHitProducer"));

  // Name of a SuperClusterCollection to make associations:
  endcapSClusterToken_ =
      consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("endcapSClusterProducer"));

  esGainToken_ = esConsumes<ESGain, ESGainRcd>();
  esMIPToGeVToken_ = esConsumes<ESMIPToGeVConstant, ESMIPToGeVConstantRcd>();
  esEEInterCalibToken_ = esConsumes<ESEEIntercalibConstants, ESEEIntercalibConstantsRcd>();
  esMissingECalibToken_ = esConsumes<ESMissingEnergyCalibration, ESMissingEnergyCalibrationRcd>();
  esChannelStatusToken_ = esConsumes<ESChannelStatus, ESChannelStatusRcd>();
  caloGeometryToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>();

  // Output collections:
  preshClusterCollectionX_ = ps.getParameter<std::string>("preshClusterCollectionX");
  preshClusterCollectionY_ = ps.getParameter<std::string>("preshClusterCollectionY");

  assocSClusterCollection_ = ps.getParameter<std::string>("assocSClusterCollection");

  produces<reco::PreshowerClusterCollection>(preshClusterCollectionX_);
  produces<reco::PreshowerClusterCollection>(preshClusterCollectionY_);
  produces<reco::SuperClusterCollection>(assocSClusterCollection_);

  float esStripECut = ps.getParameter<double>("esStripEnergyCut");
  esPhiClusterDeltaEta_ = ps.getParameter<double>("esPhiClusterDeltaEta");
  esPhiClusterDeltaPhi_ = ps.getParameter<double>("esPhiClusterDeltaPhi");

  etThresh_ = ps.getParameter<double>("etThresh");

  presh_algo = new PreshowerPhiClusterAlgo(esStripECut);
}

PreshowerPhiClusterProducer::~PreshowerPhiClusterProducer() { delete presh_algo; }

void PreshowerPhiClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<EcalRecHitCollection> pRecHits;
  edm::Handle<reco::SuperClusterCollection> pSuperClusters;

  // get the ECAL geometry:
  edm::ESHandle<CaloGeometry> geoHandle = es.getHandle(caloGeometryToken_);

  // retrieve ES-EE intercalibration constants and channel status
  set(es);
  const ESChannelStatus* channelStatus = esChannelStatus_.product();

  const CaloSubdetectorGeometry* geometry_p = (geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower));

  // create unique_ptr to a PreshowerClusterCollection
  auto clusters_p1 = std::make_unique<reco::PreshowerClusterCollection>();
  auto clusters_p2 = std::make_unique<reco::PreshowerClusterCollection>();
  // create new collection of corrected super clusters
  auto superclusters_p = std::make_unique<reco::SuperClusterCollection>();

  // fetch the product (pSuperClusters)
  evt.getByToken(endcapSClusterToken_, pSuperClusters);
  const reco::SuperClusterCollection* SClusts = pSuperClusters.product();

  // fetch the product (RecHits)
  evt.getByToken(preshHitToken_, pRecHits);
  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product();  // EcalRecHitCollection hit_collection = *rhcHandle;

  LogTrace("EcalClusters") << "PreshowerPhiClusterProducerInfo: ### Total # of preshower RecHits: " << rechits->size();

  // make the map of rechits:
  std::map<DetId, EcalRecHit> rechits_map;
  EcalRecHitCollection::const_iterator it;
  for (it = rechits->begin(); it != rechits->end(); it++) {
    // remove bad ES rechits
    if (it->recoFlag() == 1 || it->recoFlag() == 14 || (it->recoFlag() <= 10 && it->recoFlag() >= 5))
      continue;
    //Make the map of DetID, EcalRecHit pairs
    rechits_map.insert(std::make_pair(it->id(), *it));
  }
  // The set of used DetID's for a given event:
  std::set<DetId> used_strips;
  used_strips.clear();
  LogTrace("EcalClusters") << "PreshowerPhiClusterProducerInfo: ### rechits_map of size " << rechits_map.size()
                           << " was created!";

  reco::PreshowerClusterCollection clusters1, clusters2;  // output collection of corrected PCs
  reco::SuperClusterCollection new_SC;                    // output collection of corrected SCs

  //make cycle over super clusters
  reco::SuperClusterCollection::const_iterator it_super;
  int isc = 0;
  for (it_super = SClusts->begin(); it_super != SClusts->end(); ++it_super) {
    float e1 = 0;
    float e2 = 0;
    float deltaE = 0;

    reco::CaloClusterPtrVector new_BC;
    ++isc;
    LogTrace("EcalClusters") << " superE = " << it_super->energy() << " superETA = " << it_super->eta()
                             << " superPHI = " << it_super->phi();

    //cout<<"=== new SC ==="<<endl;
    //cout<<"superE = "<<it_super->energy()<<" superETA = "<<it_super->eta()<<" superPHI = "<<it_super->phi()<<endl;

    int nBC = 0;
    int condP1 = 1;  // 0: dead channel; 1: active channel
    int condP2 = 1;
    float maxDeltaPhi = 0;
    float minDeltaPhi = 0;
    float refPhi = 0;

    reco::CaloCluster_iterator bc_iter = it_super->clustersBegin();
    for (; bc_iter != it_super->clustersEnd(); ++bc_iter) {
      if (nBC == 0) {
        refPhi = (*bc_iter)->phi();
      } else {
        if (reco::deltaPhi((*bc_iter)->phi(), refPhi) > 0 && reco::deltaPhi((*bc_iter)->phi(), refPhi) > maxDeltaPhi)
          maxDeltaPhi = reco::deltaPhi((*bc_iter)->phi(), refPhi);
        if (reco::deltaPhi((*bc_iter)->phi(), refPhi) < 0 && reco::deltaPhi((*bc_iter)->phi(), refPhi) < minDeltaPhi)
          minDeltaPhi = reco::deltaPhi((*bc_iter)->phi(), refPhi);
        //cout<<"delta phi : "<<reco::deltaPhi((*bc_iter)->phi(), refPhi)<<endl;
      }
      //cout<<"BC : "<<nBC<<" "<<(*bc_iter)->energy()<<" "<<(*bc_iter)->eta()<<" "<<(*bc_iter)->phi()<<endl;
      nBC++;
    }
    maxDeltaPhi += esPhiClusterDeltaPhi_;
    minDeltaPhi -= esPhiClusterDeltaPhi_;

    nBC = 0;
    for (bc_iter = it_super->clustersBegin(); bc_iter != it_super->clustersEnd(); ++bc_iter) {
      if (geometry_p) {
        // Get strip position at intersection point of the line EE - Vertex:
        double X = (*bc_iter)->x();
        double Y = (*bc_iter)->y();
        double Z = (*bc_iter)->z();
        const GlobalPoint point(X, Y, Z);

        DetId tmp1 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 1);
        DetId tmp2 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 2);
        ESDetId strip1 = (tmp1 == DetId(0)) ? ESDetId(0) : ESDetId(tmp1);
        ESDetId strip2 = (tmp2 == DetId(0)) ? ESDetId(0) : ESDetId(tmp2);

        if (nBC == 0) {
          if (strip1 != ESDetId(0) && strip2 != ESDetId(0)) {
            ESChannelStatusMap::const_iterator status_p1 = channelStatus->getMap().find(strip1);
            ESChannelStatusMap::const_iterator status_p2 = channelStatus->getMap().find(strip2);
            if (status_p1->getStatusCode() == 1)
              condP1 = 0;
            if (status_p2->getStatusCode() == 1)
              condP2 = 0;
          } else if (strip1 == ESDetId(0)) {
            condP1 = 0;
          } else if (strip2 == ESDetId(0)) {
            condP2 = 0;
          }

          //cout<<"starting cluster : "<<maxDeltaPhi<<" "<<minDeltaPhi<<" "<<esPhiClusterDeltaEta_<<endl;
          //cout<<"do plane 1 === "<<strip1.zside()<<" "<<strip1.plane()<<" "<<strip1.six()<<" "<<strip1.siy()<<" "<<strip1.strip()<<endl;
          // Get a vector of ES clusters (found by the PreshSeeded algorithm) associated with a given EE basic cluster.
          reco::PreshowerCluster cl1 = presh_algo->makeOneCluster(
              strip1, &used_strips, &rechits_map, geometry_p, esPhiClusterDeltaEta_, minDeltaPhi, maxDeltaPhi);
          cl1.setBCRef(*bc_iter);
          clusters1.push_back(cl1);
          e1 += cl1.energy();

          //cout<<"do plane 2 === "<<strip2.zside()<<" "<<strip2.plane()<<" "<<strip2.six()<<" "<<strip2.siy()<<" "<<strip2.strip()<<endl;
          reco::PreshowerCluster cl2 = presh_algo->makeOneCluster(
              strip2, &used_strips, &rechits_map, geometry_p, esPhiClusterDeltaEta_, minDeltaPhi, maxDeltaPhi);
          cl2.setBCRef(*bc_iter);
          clusters2.push_back(cl2);
          e2 += cl2.energy();
        }
      }

      new_BC.push_back(*bc_iter);
      nBC++;
    }  // end of cycle over BCs

    LogTrace("EcalClusters") << " For SC #" << isc - 1 << ", containing " << it_super->clustersSize()
                             << " basic clusters, PreshowerPhiClusterAlgo made " << clusters1.size()
                             << " in X plane and " << clusters2.size() << " in Y plane "
                             << " preshower clusters ";

    // update energy of the SuperCluster
    if (e1 + e2 > 1.0e-10) {
      e1 = e1 / mip_;  // GeV to #MIPs
      e2 = e2 / mip_;

      if (condP1 == 1 && condP2 == 1) {
        deltaE = gamma0_ * (e1 + alpha0_ * e2);
      } else if (condP1 == 1 && condP2 == 0) {
        deltaE = gamma1_ * (e1 + alpha1_ * e2);
      } else if (condP1 == 0 && condP2 == 1) {
        deltaE = gamma2_ * (e1 + alpha2_ * e2);
      } else if (condP1 == 0 && condP2 == 0) {
        deltaE = gamma3_ * (e1 + alpha3_ * e2);
      }
    }

    //corrected Energy
    float E = it_super->energy() + deltaE;

    LogTrace("EcalClusters") << " Creating corrected SC ";

    reco::SuperCluster sc(E, it_super->position(), it_super->seed(), new_BC, deltaE);
    sc.setPreshowerEnergyPlane1(e1 * mip_);
    sc.setPreshowerEnergyPlane2(e2 * mip_);
    if (condP1 == 1 && condP2 == 1)
      sc.setPreshowerPlanesStatus(0);
    else if (condP1 == 1 && condP2 == 0)
      sc.setPreshowerPlanesStatus(1);
    else if (condP1 == 0 && condP2 == 1)
      sc.setPreshowerPlanesStatus(2);
    else if (condP1 == 0 && condP2 == 0)
      sc.setPreshowerPlanesStatus(3);

    if (etThresh_ > 0) {  // calling postion().theta can be expensive
      if (sc.energy() * sin(sc.position().theta()) > etThresh_)
        new_SC.push_back(sc);
    } else {
      new_SC.push_back(sc);
    }

  }  // end of cycle over SCs

  // copy the preshower clusters into collections and put in the Event:
  clusters_p1->assign(clusters1.begin(), clusters1.end());
  clusters_p2->assign(clusters2.begin(), clusters2.end());
  // put collection of preshower clusters to the event
  evt.put(std::move(clusters_p1), preshClusterCollectionX_);
  evt.put(std::move(clusters_p2), preshClusterCollectionY_);
  LogTrace("EcalClusters") << "Preshower clusters added to the event";

  // put collection of corrected super clusters to the event
  superclusters_p->assign(new_SC.begin(), new_SC.end());
  evt.put(std::move(superclusters_p), assocSClusterCollection_);
  LogTrace("EcalClusters") << "Corrected SClusters added to the event";
}

void PreshowerPhiClusterProducer::set(const edm::EventSetup& es) {
  esgain_ = es.getHandle(esGainToken_);
  const ESGain* gain = esgain_.product();

  double ESGain = gain->getESGain();

  esMIPToGeV_ = es.getHandle(esMIPToGeVToken_);
  const ESMIPToGeVConstant* mipToGeV = esMIPToGeV_.product();

  mip_ = (ESGain == 1) ? mipToGeV->getESValueLow() : mipToGeV->getESValueHigh();

  esChannelStatus_ = es.getHandle(esChannelStatusToken_);

  esEEInterCalib_ = es.getHandle(esEEInterCalibToken_);
  const ESEEIntercalibConstants* esEEInterCalib = esEEInterCalib_.product();

  // both planes work
  gamma0_ = (ESGain == 1) ? 0.02 : esEEInterCalib->getGammaHigh0();
  alpha0_ = (ESGain == 1) ? esEEInterCalib->getAlphaLow0() : esEEInterCalib->getAlphaHigh0();

  // only first plane works
  gamma1_ = (ESGain == 1) ? (0.02 * esEEInterCalib->getGammaLow1()) : esEEInterCalib->getGammaHigh1();
  alpha1_ = (ESGain == 1) ? esEEInterCalib->getAlphaLow1() : esEEInterCalib->getAlphaHigh1();

  // only second plane works
  gamma2_ = (ESGain == 1) ? (0.02 * esEEInterCalib->getGammaLow2()) : esEEInterCalib->getGammaHigh2();
  alpha2_ = (ESGain == 1) ? esEEInterCalib->getAlphaLow2() : esEEInterCalib->getAlphaHigh2();

  // both planes do not work
  gamma3_ = (ESGain == 1) ? 0.02 : esEEInterCalib->getGammaHigh3();
  alpha3_ = (ESGain == 1) ? esEEInterCalib->getAlphaLow3() : esEEInterCalib->getAlphaHigh3();

  esMissingECalib_ = es.getHandle(esMissingECalibToken_);
  const ESMissingEnergyCalibration* esMissingECalib = esMissingECalib_.product();

  // |eta| < 1.9
  aEta_[0] = esMissingECalib->getConstAEta0();
  bEta_[0] = esMissingECalib->getConstBEta0();

  // 1.9 < |eta| < 2.1
  aEta_[1] = esMissingECalib->getConstAEta1();
  bEta_[1] = esMissingECalib->getConstBEta1();

  // 2.1 < |eta| < 2.3
  aEta_[2] = esMissingECalib->getConstAEta2();
  bEta_[2] = esMissingECalib->getConstBEta2();

  // 2.3 < |eta| < 2.5
  aEta_[3] = esMissingECalib->getConstAEta3();
  bEta_[3] = esMissingECalib->getConstBEta3();
}
