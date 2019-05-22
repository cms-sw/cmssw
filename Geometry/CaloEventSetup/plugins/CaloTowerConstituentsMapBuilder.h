// -*- C++ -*-
//
// Package:    CaloTowerConstituentsMapBuilder
// Class:      CaloTowerConstituentsMapBuilder
// 
/**\class CaloTowerConstituentsMapBuilder CaloTowerConstituentsMapBuilder.h tmp/CaloTowerConstituentsMapBuilder/interface/CaloTowerConstituentsMapBuilder.h

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
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class CaloTowerConstituentsMapBuilder : public edm::ESProducer {
public:
  CaloTowerConstituentsMapBuilder(const edm::ParameterSet&);
  ~CaloTowerConstituentsMapBuilder() override;

  typedef std::unique_ptr<CaloTowerConstituentsMap> ReturnType;

  ReturnType produce(const CaloGeometryRecord&);
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  void parseTextMap(const std::string& filename,CaloTowerConstituentsMap& theMap);
  void assignEEtoHE(const CaloGeometry* geometry, CaloTowerConstituentsMap& theMap, const CaloTowerTopology * cttopo);
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcaltopoToken_;
  edm::ESGetToken<CaloTowerTopology, HcalRecNumberingRecord> cttopoToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  std::string mapFile_;
  bool mapAuto_, skipHE_;
};

