// -*- C++ -*-
//
// Package:    EcalTrigTowerConstituentsMapBuilder
// Class:      EcalTrigTowerConstituentsMapBuilder
//
/**\class EcalTrigTowerConstituentsMapBuilder EcalTrigTowerConstituentsMapBuilder.h tmp/EcalTrigTowerConstituentsMapBuilder/interface/EcalTrigTowerConstituentsMapBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Paolo Meridiani
//
//

#ifndef Geometry_CaloEventSetup_EcalTrigTowerConstituentsMapBuilder
#define Geometry_CaloEventSetup_EcalTrigTowerConstituentsMapBuilder

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

//
// class decleration
//

class EcalTrigTowerConstituentsMapBuilder : public edm::ESProducer {
public:
  EcalTrigTowerConstituentsMapBuilder(const edm::ParameterSet&);
  ~EcalTrigTowerConstituentsMapBuilder() override;

  typedef std::unique_ptr<EcalTrigTowerConstituentsMap> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  void parseTextMap(const std::string& filename, EcalTrigTowerConstituentsMap& theMap);
  std::string mapFile_;
  // ----------member data ---------------------------
};

#endif
