#ifndef Geometry_TrackerGeometryBuilder_TrackerDigiGeometryESModule_H
#define Geometry_TrackerGeometryBuilder_TrackerDigiGeometryESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"

#include <memory>

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

class  TrackerDigiGeometryESModule: public edm::ESProducer{
 public:
  TrackerDigiGeometryESModule(const edm::ParameterSet & p);
  ~TrackerDigiGeometryESModule() override; 
  std::unique_ptr<TrackerGeometry> produce(const TrackerDigiGeometryRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 private:
  /// Called when geometry description changes
  const std::string alignmentsLabel_;
  const std::string myLabel_;

  edm::ESGetToken<GeometricDet,IdealGeometryRecord> geometricDetToken_;
  edm::ESGetToken<TrackerTopology,TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<PTrackerParameters,PTrackerParametersRcd> trackerParamsToken_;

  edm::ESGetToken<Alignments,GlobalPositionRcd> globalAlignmentToken_;
  edm::ESGetToken<Alignments,TrackerAlignmentRcd> trackerAlignmentToken_;
  edm::ESGetToken<AlignmentErrorsExtended, TrackerAlignmentErrorExtendedRcd> alignmentErrorsToken_;
  edm::ESGetToken<AlignmentSurfaceDeformations, TrackerSurfaceDeformationRcd> deformationsToken_;

  const bool applyAlignment_; // Switch to apply alignment corrections
};


#endif
