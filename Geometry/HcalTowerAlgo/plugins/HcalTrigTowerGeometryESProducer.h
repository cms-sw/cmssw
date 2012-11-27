#ifndef HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H
# define HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H

# include "boost/shared_ptr.hpp"

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
  virtual ~HcalTrigTowerGeometryESProducer( void );

  boost::shared_ptr<HcalTrigTowerGeometry> produce( const CaloGeometryRecord & );

private:
  bool useFullGranularityHF_;
  boost::shared_ptr<HcalTrigTowerGeometry> m_hcalTrigTowerGeom;
};

#endif // HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H
