
#include "CSCGeometryESModule.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilder.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"

#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include <memory>

using namespace edm;

CSCGeometryESModule::CSCGeometryESModule(const edm::ParameterSet & p)
  : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
    myLabel_(p.getParameter<std::string>("appendToDataLabel"))
{

  // Choose wire geometry modelling
  // We now _require_ some wire geometry specification in the CSCOrcaSpec.xml file
  // in the DDD Geometry.
  // Default as of transition to CMSSW is to use real values.
  // Alternative is to use pseudo-values which match reasonably closely
  // the calculated geometry values used up to and including ORCA_8_8_1.
  // (This was the default in ORCA.)

  useRealWireGeometry =   p.getParameter<bool>("useRealWireGeometry");

  // Suppress strips altogether in ME1a region of ME11?

  useOnlyWiresInME1a =    p.getParameter<bool>("useOnlyWiresInME1a");

  // Allow strips in ME1a region of ME11 but gang them?
  // Default is now to treat ME1a with ganged strips (e.g. in clusterizer)

  useGangedStripsInME1a = p.getParameter<bool>("useGangedStripsInME1a");

  if ( useGangedStripsInME1a ) useOnlyWiresInME1a = false; // override possible inconsistentcy

  // Use the backed-out offsets that correct the CTI
  useCentreTIOffsets = p.getParameter<bool>("useCentreTIOffsets"); 

  // Debug printout etc. in CSCGeometry etc.

  debugV = p.getUntrackedParameter<bool>("debugV", false);

  // Find out if using the DDD or CondDB Geometry source.
  useDDD_ = p.getParameter<bool>("useDDD");

  // Feed these value to where I need them
  applyAlignment_ = p.getParameter<bool>("applyAlignment");


  edm::LogInfo("Geometry") << "@SUB=CSCGeometryESModule" 
			   << "Label '" << myLabel_ << "' "
			   << (applyAlignment_ ? "looking for" : "IGNORING")
			   << " alignment labels '" << alignmentsLabel_ << "'.";
  setWhatProduced(this);
}


CSCGeometryESModule::~CSCGeometryESModule(){}


std::shared_ptr<CSCGeometry> CSCGeometryESModule::produce(const MuonGeometryRecord& record) {

  auto host = holder_.makeOrGet([this]() {
    return new HostType(debugV,
                        useGangedStripsInME1a,
                        useOnlyWiresInME1a,
                        useRealWireGeometry,
                        useCentreTIOffsets);
  });

  initCSCGeometry_(record, host);

  // Called whenever the alignments or alignment errors change

  if ( applyAlignment_ ) {
    // applyAlignment_ is scheduled for removal. 
    // Ideal geometry obtained by using 'fake alignment' (with applyAlignment_ = true)
    edm::ESHandle<Alignments> globalPosition;
    record.getRecord<GlobalPositionRcd>().get(alignmentsLabel_, globalPosition);
    edm::ESHandle<Alignments> alignments;
    record.getRecord<CSCAlignmentRcd>().get(alignmentsLabel_, alignments);
    edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
    record.getRecord<CSCAlignmentErrorExtendedRcd>().get(alignmentsLabel_,  alignmentErrors);
    // Only apply alignment if values exist
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogInfo("Config") << "@SUB=CSCGeometryRecord::produce"
			     << "Alignment(Error)s and global position (label '"
			     << alignmentsLabel_ << "') empty: Geometry producer (label "
			     << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<CSCGeometry>( &(*host), &(*alignments), &(*alignmentErrors),
	                    align::DetectorGlobalPosition(*globalPosition, DetId(DetId::Muon)) );
    }
  }
  return host; // automatically converts to std::shared_ptr<CSCGeometry>
}


void CSCGeometryESModule::initCSCGeometry_( const MuonGeometryRecord& record, std::shared_ptr<HostType>& host)
{
  if ( useDDD_ ) {

    host->ifRecordChanges<MuonNumberingRecord>(record,
                                               [this, &host, &record](auto const& rec) {
      host->clear();
      edm::ESTransientHandle<DDCompactView> cpv;
      edm::ESHandle<MuonDDDConstants> mdc;
      record.getRecord<IdealGeometryRecord>().get(cpv);
      rec.get(mdc);
      CSCGeometryBuilderFromDDD builder;
      builder.build(*host, &(*cpv), *mdc);
    });
  } else {
    bool recreateGeometry = false;

    host->ifRecordChanges<CSCRecoGeometryRcd>(record,
                                               [&recreateGeometry](auto const& rec) {
      recreateGeometry = true;
    });

    host->ifRecordChanges<CSCRecoDigiParametersRcd>(record,
                                               [&recreateGeometry](auto const& rec) {
      recreateGeometry = true;
    });

    if (recreateGeometry) {
      host->clear();
      edm::ESHandle<RecoIdealGeometry> rig;
      edm::ESHandle<CSCRecoDigiParameters> rdp;
      record.getRecord<CSCRecoGeometryRcd>().get(rig);
      record.getRecord<CSCRecoDigiParametersRcd>().get(rdp);
      CSCGeometryBuilder cscgb;
      cscgb.build(*host, *rig, *rdp);
    }
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(CSCGeometryESModule);
