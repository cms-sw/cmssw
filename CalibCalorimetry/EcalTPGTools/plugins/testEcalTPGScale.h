#ifndef TESTECALTPGSCALE_H
#define TESTECALTPGSCALE_H

//Author: Pascal Paganini - LLR
//Date: 2006/07/10 15:58:06 $

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/EcalEndcapGeometryRecord.h"
#include "Geometry/Records/interface/EcalBarrelGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class CaloSubdetectorGeometry;

class testEcalTPGScale : public edm::EDAnalyzer {
public:
  explicit testEcalTPGScale(edm::ParameterSet const& pSet);
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void beginJob() override;

private:
  const CaloSubdetectorGeometry* theEndcapGeometry_;
  const CaloSubdetectorGeometry* theBarrelGeometry_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalEndcapGeometryRecord> endcapGeomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometryRecord> barrelGeomToken_;
  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> eTTmapToken_;
  EcalTPGScale::Tokens tokens_;
};
#endif
