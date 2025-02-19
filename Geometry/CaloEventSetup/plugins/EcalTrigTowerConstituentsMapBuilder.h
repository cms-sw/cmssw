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
// $Id: EcalTrigTowerConstituentsMapBuilder.h,v 1.1 2007/04/15 23:16:28 wmtan Exp $
//
//

#ifndef Geometry_CaloEventSetup_EcalTrigTowerConstituentsMapBuilder
#define Geometry_CaloEventSetup_EcalTrigTowerConstituentsMapBuilder

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

//
// class decleration
//

class EcalTrigTowerConstituentsMapBuilder : public edm::ESProducer {
   public:
  EcalTrigTowerConstituentsMapBuilder(const edm::ParameterSet&);
  ~EcalTrigTowerConstituentsMapBuilder();

  typedef std::auto_ptr<EcalTrigTowerConstituentsMap> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  void parseTextMap(const std::string& filename,EcalTrigTowerConstituentsMap& theMap);
  std::string mapFile_;
      // ----------member data ---------------------------
};

#endif
