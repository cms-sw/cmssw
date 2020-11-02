// -*- C++ -*-
//
// Package:    DetectorDescription/MuonOffsetESProducer
// Class:      MuonOffsetESProducer
//
/**\class MuonOffsetESProducer

 Description: Produce offsets and tags for Muon volume copy numbers

 Implementation:
     The constants are defined in XML as SpecPars
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Fri, 16 Oct 2020 09:10:32 GMT
//
//

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/GeometryObjects/interface/MuonOffsetMap.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonOffsetFromDD.h"

#include <memory>
#include <string>
#include <vector>

//#define EDM_ML_DEBUG

class MuonOffsetESProducer : public edm::ESProducer {
public:
  MuonOffsetESProducer(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MuonOffsetMap>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  ReturnType produce(const IdealGeometryRecord&);

private:
  const bool fromDD4Hep_;
  const std::vector<std::string> names_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
};

MuonOffsetESProducer::MuonOffsetESProducer(const edm::ParameterSet& iConfig)
    : fromDD4Hep_(iConfig.getParameter<bool>("fromDD4Hep")),
      names_(iConfig.getParameter<std::vector<std::string>>("names")) {
  auto cc = setWhatProduced(this);
  if (fromDD4Hep_) {
    cpvTokenDD4Hep_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  } else {
    cpvTokenDDD_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonOffsetESProducer::MuonOffsetESProducer called with dd4hep: " << fromDD4Hep_;
#endif
}

void MuonOffsetESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names = {"MuonCommonNumbering",
                                    "MuonBarrel",
                                    "MuonEndcap",
                                    "MuonBarrelWheels",
                                    "MuonBarrelStation1",
                                    "MuonBarrelStation2",
                                    "MuonBarrelStation3",
                                    "MuonBarrelStation4",
                                    "MuonBarrelSuperLayer",
                                    "MuonBarrelLayer",
                                    "MuonBarrelWire",
                                    "MuonRpcPlane1I",
                                    "MuonRpcPlane1O",
                                    "MuonRpcPlane2I",
                                    "MuonRpcPlane2O",
                                    "MuonRpcPlane3S",
                                    "MuonRpcPlane4",
                                    "MuonRpcChamberLeft",
                                    "MuonRpcChamberMiddle",
                                    "MuonRpcChamberRight",
                                    "MuonRpcEndcap1",
                                    "MuonRpcEndcap2",
                                    "MuonRpcEndcap3",
                                    "MuonRpcEndcap4",
                                    "MuonRpcEndcapSector",
                                    "MuonRpcEndcapChamberB1",
                                    "MuonRpcEndcapChamberB2",
                                    "MuonRpcEndcapChamberB3",
                                    "MuonRpcEndcapChamberC1",
                                    "MuonRpcEndcapChamberC2",
                                    "MuonRpcEndcapChamberC3",
                                    "MuonRpcEndcapChamberE1",
                                    "MuonRpcEndcapChamberE2",
                                    "MuonRpcEndcapChamberE3",
                                    "MuonRpcEndcapChamberF1",
                                    "MuonRpcEndcapChamberF2",
                                    "MuonRpcEndcapChamberF3",
                                    "MuonEndcapStation1",
                                    "MuonEndcapStation2",
                                    "MuonEndcapStation3",
                                    "MuonEndcapStation4",
                                    "MuonEndcapSubrings",
                                    "MuonEndcapSectors",
                                    "MuonEndcapLayers",
                                    "MuonEndcapRing1",
                                    "MuonEndcapRing2",
                                    "MuonEndcapRing3",
                                    "MuonEndcapRingA",
                                    "MuonGEMEndcap",
                                    "MuonGEMSector",
                                    "MuonGEMChamber"};
  desc.add<bool>("fromDD4Hep", false);
  desc.add<std::vector<std::string>>("names", names);
  descriptions.add("muonOffsetESProducer", desc);
}

MuonOffsetESProducer::ReturnType MuonOffsetESProducer::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonOffsetESProducer::produce(const IdealGeometryRecord& iRecord)";
#endif

  auto ptp = std::make_unique<MuonOffsetMap>();
  MuonOffsetFromDD builder(names_);

  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonGeom") << "MuonOffsetESProducer::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(&(*cpv), *ptp);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonGeom") << "MuonOffsetESProducer::Try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(&(*cpv), *ptp);
  }
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonOffsetESProducer);
