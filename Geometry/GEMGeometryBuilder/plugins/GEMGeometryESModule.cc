/** \file
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometryBuilder/plugins/GEMGeometryESModule.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromCondDB.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include <memory>

using namespace edm;

GEMGeometryESModule::GEMGeometryESModule(const edm::ParameterSet & p)
  : useDDD(p.getParameter<bool>("useDDD")),
    applyAlignment(p.getParameter<bool>("applyAlignment")),
    alignmentsLabel(p.getParameter<std::string>("alignmentsLabel"))
{
  setWhatProduced(this);
}

GEMGeometryESModule::~GEMGeometryESModule(){}

std::unique_ptr<GEMGeometry>
GEMGeometryESModule::produce(const MuonGeometryRecord & record) 
{
  auto gemGeometry = std::make_unique<GEMGeometry>();

  if( useDDD ) {
    edm::ESTransientHandle<DDCompactView> cpv;
    record.getRecord<IdealGeometryRecord>().get(cpv);
    edm::ESHandle<MuonDDDConstants> mdc;
    record.getRecord<MuonNumberingRecord>().get(mdc);
    GEMGeometryBuilderFromDDD builder;
    builder.build(*gemGeometry, &(*cpv), *mdc);
  } else {
    edm::ESHandle<RecoIdealGeometry> riggem;
    record.getRecord<GEMRecoGeometryRcd>().get(riggem);
    GEMGeometryBuilderFromCondDB builder;
    builder.build(*gemGeometry, *riggem);
  }

  if (applyAlignment) {
    edm::ESHandle<Alignments> globalPosition;
    record.getRecord<GlobalPositionRcd>().get(alignmentsLabel, globalPosition);
    edm::ESHandle<Alignments> alignments;
    record.getRecord<GEMAlignmentRcd>().get(alignmentsLabel, alignments);
    edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
    record.getRecord<GEMAlignmentErrorExtendedRcd>().get(alignmentsLabel, alignmentErrors);
    
    // No alignment records, assume ideal geometry is wanted
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogInfo("Config") << "@SUB=GEMGeometryRecord::produce"
			     << "Alignment(Error)s and global position (label '"
			     << alignmentsLabel << "') empty: it is assumed fake and will not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<GEMGeometry>(&(*gemGeometry), &(*alignments), &(*alignmentErrors),
					   align::DetectorGlobalPosition(*globalPosition, DetId(DetId::Muon)));
    }
  }

  return gemGeometry;  
}

DEFINE_FWK_EVENTSETUP_MODULE(GEMGeometryESModule);
