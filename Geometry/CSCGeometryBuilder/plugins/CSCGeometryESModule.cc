
#include "CSCGeometryESModule.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include <memory>

using namespace edm;

CSCGeometryESModule::CSCGeometryESModule(const edm::ParameterSet & p){

  setWhatProduced(this, dependsOn(&CSCGeometryESModule::geometryCallback_) );

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

  // Switch to apply the alignment corrections

  applyAlignment_ = p.getUntrackedParameter<bool>("applyAlignment", false);

}


CSCGeometryESModule::~CSCGeometryESModule(){}


boost::shared_ptr<CSCGeometry> CSCGeometryESModule::produce(const MuonGeometryRecord& record) {

  // Called whenever the alignments or alignment errors change

  if ( applyAlignment_ ) {
    // applyAlignment_ is scheduled for removal. 
    // Ideal geometry obtained by using 'fake alignment' (with applyAlignment_ = true)
    edm::ESHandle<Alignments> globalPosition;
    record.getRecord<GlobalPositionRcd>().get( globalPosition );
    edm::ESHandle<Alignments> alignments;
    record.getRecord<CSCAlignmentRcd>().get( alignments );
    edm::ESHandle<AlignmentErrors> alignmentErrors;
    record.getRecord<CSCAlignmentErrorRcd>().get( alignmentErrors );
    // Only apply alignment if values exist
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogWarning("Config") << "@SUB=CSCGeometryBuilder::produce"
                                << "Empty Alignment(Error)s and global position, "
                                << "so nothing to apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<CSCGeometry>( &(*cscGeometry), &(*alignments), &(*alignmentErrors),
	                    align::DetectorGlobalPosition(*globalPosition, DetId(DetId::Muon)) );
    }
  }

  return cscGeometry;
}


void CSCGeometryESModule::geometryCallback_( const MuonNumberingRecord& record )
{
  // Called whenever the muon numbering (or ideal geometry) changes

  cscGeometry = boost::shared_ptr<CSCGeometry>( new CSCGeometry );

  cscGeometry->setUseRealWireGeometry( useRealWireGeometry );
  cscGeometry->setOnlyWiresInME1a( useOnlyWiresInME1a );
  cscGeometry->setGangedStripsInME1a( useGangedStripsInME1a );
  cscGeometry->setUseCentreTIOffsets( useCentreTIOffsets );
  cscGeometry->setDebugV( debugV );
  if ( debugV ) cscGeometry->queryModelling();

  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<MuonDDDConstants> mdc;
  record.getRecord<IdealGeometryRecord>().get(cpv);
  record.get( mdc );
  CSCGeometryBuilderFromDDD builder;
  //  cscGeometry = boost::shared_ptr<CSCGeometry>( builder.build( &(*cpv), *mdc ) );
  builder.build( cscGeometry, &(*cpv), *mdc );
}

DEFINE_FWK_EVENTSETUP_MODULE(CSCGeometryESModule);
