#ifndef GEOMETRY_HCALEVENTSETUP_CaloTowerTopologyEP_H
#define GEOMETRY_HCALEVENTSETUP_CaloTowerTopologyEP_H 1


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class CaloTowerTopologyEP : public edm::ESProducer {

public:
  CaloTowerTopologyEP(const edm::ParameterSet&);
  ~CaloTowerTopologyEP();

  typedef boost::shared_ptr<CaloTowerTopology> ReturnType;

  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );
    
  ReturnType produce(const HcalRecNumberingRecord&);

private:
  // ----------member data ---------------------------
  const edm::ParameterSet m_pSet;

};
#endif
