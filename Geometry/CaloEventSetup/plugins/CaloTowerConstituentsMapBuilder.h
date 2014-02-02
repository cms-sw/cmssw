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
// $Id: CaloTowerConstituentsMapBuilder.h,v 1.4 2012/10/26 09:47:48 yana Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class CaloTowerConstituentsMapBuilder : public edm::ESProducer {
public:
  CaloTowerConstituentsMapBuilder(const edm::ParameterSet&);
  ~CaloTowerConstituentsMapBuilder();

  typedef std::auto_ptr<CaloTowerConstituentsMap> ReturnType;

  ReturnType produce(const HcalRecNumberingRecord&);
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  void parseTextMap(const std::string& filename,CaloTowerConstituentsMap& theMap);
  std::string mapFile_;
};

