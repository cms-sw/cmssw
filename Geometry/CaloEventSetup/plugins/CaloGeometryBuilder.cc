// -*- C++ -*-
//
// Package:    CaloGeometryBuilder
// Class:      CaloGeometryBuilder
//
/**\class CaloGeometryBuilder CaloGeometryBuilder.h tmp/CaloGeometryBuilder/interface/CaloGeometryBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

// user include files
#include "Geometry/CaloEventSetup/plugins/CaloGeometryBuilder.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {
  template <typename Record>
  void makeToken(edm::ESConsumesCollector& cc,
                 std::vector<std::string>& list,
                 std::string const& tag,
                 edm::ESGetToken<CaloSubdetectorGeometry, Record>& token) {
    auto found = std::find(list.begin(), list.end(), tag);
    edm::LogVerbatim("CaloGeometryBuilder") << "Finds tag " << tag << " : " << (found != list.end());
    if (found != list.end()) {
      token = cc.consumesFrom<CaloSubdetectorGeometry, Record>(edm::ESInputTag{"", *found});
      list.erase(found);
    }
  }
}  // namespace

//
// member functions
//
CaloGeometryBuilder::CaloGeometryBuilder(const edm::ParameterSet& iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this, &CaloGeometryBuilder::produceAligned);

  //now do what ever other initialization is needed
  auto caloList = iConfig.getParameter<std::vector<std::string> >("SelectedCalos");
  if (caloList.empty())
    throw cms::Exception("Configuration") << "No calorimeter specified for geometry, aborting";

  makeToken(cc, caloList, HcalGeometry::producerTag(), hcalToken_);
  makeToken(cc, caloList, ZdcGeometry::producerTag(), zdcToken_);
  makeToken(cc, caloList, CastorGeometry::producerTag(), castorToken_);
  makeToken(cc, caloList, EcalBarrelGeometry::producerTag(), ecalBarrelToken_);
  makeToken(cc, caloList, EcalEndcapGeometry::producerTag(), ecalEndcapToken_);
  makeToken(cc, caloList, EcalPreshowerGeometry::producerTag(), ecalPreshowerToken_);
  makeToken(cc, caloList, CaloTowerGeometry::producerTag(), caloTowerToken_);

  // Move HGC elements to the end
  auto hgcBegin = std::partition(caloList.begin(), caloList.end(), [](std::string const& elem) {
    return elem.find(HGCalGeometry::producerTag()) == std::string::npos;
  });
  // Process HGC elements
  for (auto iter = hgcBegin; iter != caloList.end(); ++iter) {
    hgcalTokens_.emplace_back(cc.consumesFrom<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", *iter}), *iter);
  }
  // Erase HGC elements
  caloList.erase(hgcBegin, caloList.end());

  // Throw if any elements are left
  if (not caloList.empty()) {
    cms::Exception ex{"Configuration"};
    ex << "Reconstruction geometry requested for a not implemented sub-detectors:";
    for (auto const& elem : caloList) {
      ex << " " << elem;
    }
    throw ex;
  }
}

// ------------ method called to produce the data  ------------

CaloGeometryBuilder::ReturnType CaloGeometryBuilder::produceAligned(const CaloGeometryRecord& iRecord) {
  ReturnType pCalo = std::make_unique<CaloGeometry>();

  // look for HCAL parts
  // assume 'HCAL' for all of HCAL.
  if (hcalToken_.isInitialized()) {
    edm::LogVerbatim("CaloGeometryBuilder") << "Building HCAL reconstruction geometry";

    auto const& pG = iRecord.get(hcalToken_);
    pCalo->setSubdetGeometry(DetId::Hcal, HcalBarrel, &pG);
    pCalo->setSubdetGeometry(DetId::Hcal, HcalEndcap, &pG);
    pCalo->setSubdetGeometry(DetId::Hcal, HcalOuter, &pG);
    pCalo->setSubdetGeometry(DetId::Hcal, HcalForward, &pG);
  }
  if (zdcToken_.isInitialized()) {
    edm::LogVerbatim("CaloGeometryBuilder") << "Building ZDC reconstruction geometry";
    auto const& pG = iRecord.get(zdcToken_);
    pCalo->setSubdetGeometry(DetId::Calo, HcalZDCDetId::SubdetectorId, &pG);
  }
  if (castorToken_.isInitialized()) {
    edm::LogVerbatim("CaloGeometryBuilder") << "Building CASTOR reconstruction geometry";
    auto const& pG = iRecord.get(castorToken_);
    pCalo->setSubdetGeometry(DetId::Calo, HcalCastorDetId::SubdetectorId, &pG);
  }

  // look for Ecal Barrel
  if (ecalBarrelToken_.isInitialized()) {
    edm::LogVerbatim("CaloGeometryBuilder") << "Building EcalBarrel reconstruction geometry";
    auto const& pG = iRecord.get(ecalBarrelToken_);
    pCalo->setSubdetGeometry(DetId::Ecal, EcalBarrel, &pG);
  }
  // look for Ecal Endcap
  if (ecalEndcapToken_.isInitialized()) {
    edm::LogVerbatim("CaloGeometryBuilder") << "Building EcalEndcap reconstruction geometry";
    auto const& pG = iRecord.get(ecalEndcapToken_);
    pCalo->setSubdetGeometry(DetId::Ecal, EcalEndcap, &pG);
  }
  // look for Ecal Preshower
  if (ecalPreshowerToken_.isInitialized()) {
    edm::LogVerbatim("CaloGeometryBuilder") << "Building EcalPreshower reconstruction geometry";
    const auto& pG = iRecord.get(ecalPreshowerToken_);
    pCalo->setSubdetGeometry(DetId::Ecal, EcalPreshower, &pG);
  }

  // look for TOWER parts
  if (caloTowerToken_.isInitialized()) {
    edm::LogVerbatim("CaloGeometryBuilder") << "Building TOWER reconstruction geometry";
    const auto& pG = iRecord.get(caloTowerToken_);
    pCalo->setSubdetGeometry(DetId::Calo, 1, &pG);
  }

  for (auto const& hgcTokenLabel : hgcalTokens_) {
    edm::LogVerbatim("CaloGeometryBuilder") << "Building " << hgcTokenLabel.second << " reconstruction geometry";
    auto const& pHG = iRecord.get(hgcTokenLabel.first);
    const auto& topo = pHG.topology();
    pCalo->setSubdetGeometry(topo.detector(), topo.subDetector(), &pHG);
  }

  return pCalo;
}
