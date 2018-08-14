#ifndef GEOMETRY_HCALEVENTSETUP_HCALTOPOLOGYIDEALEP_H
#define GEOMETRY_HCALEVENTSETUP_HCALTOPOLOGYIDEALEP_H 1


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class HcalTopologyIdealEP : public edm::ESProducer {

public:
  HcalTopologyIdealEP(const edm::ParameterSet&);
  ~HcalTopologyIdealEP() override;

  typedef std::shared_ptr<HcalTopology> ReturnType;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    
  ReturnType produce(const HcalRecNumberingRecord&);

  void       hcalRecordCallBack( const IdealGeometryRecord& ) {}

private:
  // ----------member data ---------------------------
  std::string m_restrictions;
  bool        m_mergePosition;
  const edm::ParameterSet m_pSet;
};
#endif
