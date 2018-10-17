#ifndef HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H
# define HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H

# include <memory>

# include "FWCore/Framework/interface/ESProducer.h"
# include "Geometry/Records/interface/CaloGeometryRecord.h"
# include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HcalTrigTowerGeometryESProducer : public edm::ESProducer
{
public:
  HcalTrigTowerGeometryESProducer( const edm::ParameterSet & conf );
  ~HcalTrigTowerGeometryESProducer( void ) override;

  std::unique_ptr<HcalTrigTowerGeometry> produce( const CaloGeometryRecord & );

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
};

#endif // HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H
