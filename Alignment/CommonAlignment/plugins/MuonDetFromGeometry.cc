#include "Alignment/CommonAlignment/interface/AlignSetup.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/DataRecord/interface/CSCAlignmentErrorRcd.h"
#include "CondFormats/DataRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/DataRecord/interface/DTAlignmentRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include "Alignment/CommonAlignment/plugins/MuonDetFromGeometry.h"

MuonDetFromGeometry::MuonDetFromGeometry(const edm::ParameterSet& cfg):
  applyAlignment_( cfg.getUntrackedParameter<bool>("applyAlignment", false) )
{
}

void MuonDetFromGeometry::beginJob(const edm::EventSetup& setup)
{
  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<MuonDDDConstants> mdc;

  setup.get<IdealGeometryRecord>().get(cpv);
  setup.get<MuonNumberingRecord>().get(mdc);

  DTGeometry* muonDT = DTGeometryBuilderFromDDD().build(&*cpv, *mdc);
  boost::shared_ptr<CSCGeometry> muonCSC( new CSCGeometry );

  CSCGeometryBuilderFromDDD().build(muonCSC, &*cpv, *mdc);

  if (applyAlignment_)
  {
    GeometryAligner aligner;

    edm::ESHandle<Alignments>      values;
    edm::ESHandle<AlignmentErrors> errors;

    setup.get<DTAlignmentRcd>().get(values);
    setup.get<DTAlignmentErrorRcd>().get(errors);
    aligner.applyAlignments(muonDT, &*values, &*errors);

    setup.get<CSCAlignmentRcd>().get(values);
    setup.get<CSCAlignmentErrorRcd>().get(errors);
    aligner.applyAlignments(&*muonCSC, &*values, &*errors);
  }

  AlignSetup<DTGeometry>::get()  = *muonDT;
  AlignSetup<CSCGeometry>::get() = *muonCSC;

//   delete muonDT;  // fixme: crash on delete; who owns the object?
}
