#ifndef HCAL_TOWER_ALGO_HCAL_GEOMETRY_ES_PRODUCER_H
# define HCAL_TOWER_ALGO_HCAL_GEOMETRY_ES_PRODUCER_H

# include "FWCore/Framework/interface/ESProducer.h"
# include "FWCore/ParameterSet/interface/ParameterSet.h"
# include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
# include <boost/shared_ptr.hpp>

namespace edm {
  class ConfigurationDescriptions;
}

class HcalGeometryESProducer : public edm::ESProducer
{
public:
  HcalGeometryESProducer( const edm::ParameterSet & p );
  virtual ~HcalGeometryESProducer( void );

  boost::shared_ptr<HcalGeometry> produce( const IdealGeometryRecord & iRecord );
  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );

private:
  /// Called when geometry description changes
  boost::shared_ptr<HcalGeometry> m_hcal;
  const edm::ParameterSet m_pSet;
};

#endif // HCAL_TOWER_ALGO_HCAL_GEOMETRY_ES_PRODUCER_H
