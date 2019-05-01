#ifndef CSCGeometryBuilder_CSCGeometryESModule_h
#define CSCGeometryBuilder_CSCGeometryESModule_h

/** \class CSCGeometryESModule
 * 
 *  ESProducer for CSCGeometry in MuonGeometryRecord
 *
 *  \author Tim Cox
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include "FWCore/Framework/interface/ESProductHost.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"

#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// Alignments
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"

#include <memory>
#include <string>

class CSCGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  CSCGeometryESModule(const edm::ParameterSet& p);

  /// Destructor
  ~CSCGeometryESModule() override;

  /// Produce CSCGeometry
  std::shared_ptr<CSCGeometry> produce(const MuonGeometryRecord& record);

private:  


  using HostType = edm::ESProductHost<CSCGeometry,
                                      MuonNumberingRecord,
                                      CSCRecoGeometryRcd,
                                      CSCRecoDigiParametersRcd>;

  void initCSCGeometry_(const MuonGeometryRecord&, std::shared_ptr<HostType>& host);

  edm::ReusableObjectHolder<HostType> holder_;

  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  edm::ESGetToken<MuonDDDConstants, MuonNumberingRecord> mdcToken_;

  edm::ESGetToken<RecoIdealGeometry, CSCRecoGeometryRcd> rigToken_;
  edm::ESGetToken<CSCRecoDigiParameters, CSCRecoDigiParametersRcd> rdpToken_;

  edm::ESGetToken<Alignments, GlobalPositionRcd> globalPositionToken_;
  edm::ESGetToken<Alignments, CSCAlignmentRcd> alignmentsToken_;
  edm::ESGetToken<AlignmentErrorsExtended, CSCAlignmentErrorExtendedRcd> alignmentErrorsToken_;

  // Flags for controlling geometry modelling during build of CSCGeometry
  bool useRealWireGeometry;
  bool useOnlyWiresInME1a;
  bool useGangedStripsInME1a;
  bool useCentreTIOffsets;
  bool debugV;
  bool applyAlignment_; // Switch to apply alignment corrections
  bool useDDD_; // whether to build from DDD or DB
  const std::string alignmentsLabel_;
  const std::string myLabel_;
};
#endif
