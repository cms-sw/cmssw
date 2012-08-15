#ifndef HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H
# define HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H

# include "boost/shared_ptr.hpp"

# include "FWCore/Framework/interface/ESProducer.h"
# include "FWCore/ParameterSet/interface/ParameterSet.h"
# include "Geometry/Records/interface/CaloGeometryRecord.h"
# include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HcalTrigTowerGeometryESProducer : public edm::ESProducer
{
public:
  explicit HcalTrigTowerGeometryESProducer( const edm::ParameterSet & conf );
  virtual ~HcalTrigTowerGeometryESProducer( void );

  boost::shared_ptr<HcalTrigTowerGeometry> produce( const CaloGeometryRecord & );

  static void fillDescription( edm::ConfigurationDescriptions & descriptions );

private:

  boost::shared_ptr<HcalTrigTowerGeometry> m_hcalTrigTowerGeom;
  const edm::ParameterSet m_pSet;
};

#endif // HCAL_TOWER_ALGO_HCAL_TRIG_TOWER_GEOMETRY_ES_PRODUCER_H
