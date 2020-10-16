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

#include <memory>

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

//#define EDM_ML_DEBUG

class MuonOffsetESProducer : public edm::ESProducer {
public:
  MuonOffsetESProducer(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MuonOffsetMap>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  ReturnType produce(const IdealGeometryRecord&);

private:
  const bool fromDD4Hep_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
};

MuonOffsetESProducer::MuonOffsetESProducer(const edm::ParameterSet& iConfig)
  : fromDD4Hep_(iConfig.getParameter<bool>("fromDD4Hep")) {
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
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("muonOffsetESProducer", desc);
}

MuonOffsetESProducer::ReturnType MuonOffsetESProducer::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonOffsetESProducer::produce(const IdealGeometryRecord& iRecord)";
#endif

  auto ptp = std::make_unique<MuonOffsetMap>();
  MuonOffsetFromDD builder;

  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "CaloSimParametersESModule::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(&(*cpv), *ptp);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "CaloSimParametersESModule::Try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(&(*cpv), *ptp);
  }
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonOffsetESProducer);
