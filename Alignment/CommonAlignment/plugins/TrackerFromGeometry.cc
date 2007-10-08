#include "Alignment/CommonAlignment/interface/AlignSetup.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include "Alignment/CommonAlignment/plugins/TrackerFromGeometry.h"

TrackerFromGeometry::TrackerFromGeometry(const edm::ParameterSet& cfg):
  applyAlignment_( cfg.getUntrackedParameter<bool>("applyAlignment", false) )
{
}

void TrackerFromGeometry::beginJob(const edm::EventSetup& setup)
{
  edm::ESHandle<GeometricDet> geom;

  setup.get<IdealGeometryRecord>().get(geom);

  TrackerGeometry* tracker =
    TrackerGeomBuilderFromGeometricDet().build(&*geom);

  if (applyAlignment_)
  {
    edm::ESHandle<Alignments>      values;
    edm::ESHandle<AlignmentErrors> errors;

    setup.get<TrackerAlignmentRcd>().get(values);
    setup.get<TrackerAlignmentErrorRcd>().get(errors);

    GeometryAligner().applyAlignments(tracker, &*values, &*errors);
  }

  AlignSetup<TrackerGeometry>::get() = *tracker;

  delete tracker;
}
