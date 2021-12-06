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

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

//
// class decleration
//

class CaloGeometryBuilder : public edm::ESProducer {
public:
  using ReturnType = std::unique_ptr<CaloGeometry>;

  CaloGeometryBuilder(const edm::ParameterSet& iConfig);

  ~CaloGeometryBuilder() override{};

  ReturnType produceAligned(const CaloGeometryRecord& iRecord);

private:
  // ----------member data ---------------------------

  edm::ESGetToken<CaloSubdetectorGeometry, HcalGeometryRecord> hcalToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, ZDCGeometryRecord> zdcToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, CastorGeometryRecord> castorToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometryRecord> ecalBarrelToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalEndcapGeometryRecord> ecalEndcapToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalPreshowerGeometryRecord> ecalPreshowerToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, CaloTowerGeometryRecord> caloTowerToken_;
  std::vector<std::pair<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>, std::string>> hgcalTokens_;
};
