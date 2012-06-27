#ifndef Geometry_TrackerGeometryBuilder_TrackerDigiGeometryESModule_H
#define Geometry_TrackerGeometryBuilder_TrackerDigiGeometryESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include <boost/shared_ptr.hpp>

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

class  TrackerDigiGeometryESModule: public edm::ESProducer{
 public:
  TrackerDigiGeometryESModule(const edm::ParameterSet & p);
  virtual ~TrackerDigiGeometryESModule(); 
  boost::shared_ptr<TrackerGeometry> produce(const TrackerDigiGeometryRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 private:
  /// Called when geometry description changes
  boost::shared_ptr<TrackerGeometry> _tracker;
  const std::string alignmentsLabel_;
  const std::string myLabel_;
  int m_ROWS_PER_ROC;
  int m_COLS_PER_ROC;
  int m_BIG_PIX_PER_ROC_X;
  int m_BIG_PIX_PER_ROC_Y;
  int m_ROCS_X;
  int m_ROCS_Y;
  bool m_upgradeGeometry;
  bool applyAlignment_; // Switch to apply alignment corrections
  bool fromDDD_;
};


#endif




