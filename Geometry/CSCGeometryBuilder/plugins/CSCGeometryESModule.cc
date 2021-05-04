/*
// \class CSCGeometryESModule
//
//  Description: CSC ESModule for DD4hep
//              
//
// \author Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//         Created:  Thu, 05 March 2020 
//         Modified: Thu, 04 June 2020, following what made in PR #30047               
//
//         Original author: Tim Cox
*/

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilder.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"

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

  /// Creation of configuration file
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  /// Produce CSCGeometry
  std::shared_ptr<CSCGeometry> produce(const MuonGeometryRecord& record);

private:
  using HostType = edm::ESProductHost<CSCGeometry, IdealGeometryRecord, CSCRecoGeometryRcd, CSCRecoDigiParametersRcd>;

  void initCSCGeometry_(const MuonGeometryRecord&, std::shared_ptr<HostType>& host);

  edm::ReusableObjectHolder<HostType> holder_;
  // DDD
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> mdcToken_;
  //dd4hep
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokendd4hep_;

  edm::ESGetToken<RecoIdealGeometry, CSCRecoGeometryRcd> rigToken_;
  edm::ESGetToken<CSCRecoDigiParameters, CSCRecoDigiParametersRcd> rdpToken_;

  edm::ESGetToken<Alignments, GlobalPositionRcd> globalPositionToken_;
  edm::ESGetToken<Alignments, CSCAlignmentRcd> alignmentsToken_;
  edm::ESGetToken<AlignmentErrorsExtended, CSCAlignmentErrorExtendedRcd> alignmentErrorsToken_;

  // Flags for controlling geometry modelling during build of CSCGeometry
  bool useRealWireGeometry_;
  bool useOnlyWiresInME1a_;
  bool useGangedStripsInME1a_;
  bool useCentreTIOffsets_;
  bool debugV_;
  bool applyAlignment_;  // Switch to apply alignment corrections
  bool fromDDD_;         // whether to build from DDD or DB
  bool fromDD4hep_;
  const std::string alignmentsLabel_;
  const std::string myLabel_;
};

using namespace edm;

CSCGeometryESModule::CSCGeometryESModule(const edm::ParameterSet& p)
    : fromDDD_(p.getParameter<bool>("fromDDD")),
      fromDD4hep_(p.getParameter<bool>("fromDD4hep")),
      alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
      myLabel_(p.getParameter<std::string>("appendToDataLabel")) {
  auto cc = setWhatProduced(this);

  // Choose wire geometry modelling
  // We now _require_ some wire geometry specification in the CSCOrcaSpec.xml file
  // in the DDD Geometry.
  // Default as of transition to CMSSW is to use real values.
  // Alternative is to use pseudo-values which match reasonably closely
  // the calculated geometry values used up to and including ORCA_8_8_1.
  // (This was the default in ORCA.)

  useRealWireGeometry_ = p.getParameter<bool>("useRealWireGeometry");

  // Suppress strips altogether in ME1a region of ME11?

  useOnlyWiresInME1a_ = p.getParameter<bool>("useOnlyWiresInME1a");

  // Allow strips in ME1a region of ME11 but gang them?
  // Default is now to treat ME1a with ganged strips (e.g. in clusterizer)

  useGangedStripsInME1a_ = p.getParameter<bool>("useGangedStripsInME1a");

  if (useGangedStripsInME1a_)
    useOnlyWiresInME1a_ = false;  // override possible inconsistentcy

  // Use the backed-out offsets that correct the CTI
  useCentreTIOffsets_ = p.getParameter<bool>("useCentreTIOffsets");

  // Debug printout etc. in CSCGeometry etc.

  debugV_ = p.getUntrackedParameter<bool>("debugV", false);

  if (fromDDD_) {
    cpvToken_ = cc.consumes();
    mdcToken_ = cc.consumes();
  } else if (fromDD4hep_) {
    cpvTokendd4hep_ = cc.consumes();
    mdcToken_ = cc.consumes();
  } else {
    rigToken_ = cc.consumesFrom<RecoIdealGeometry, CSCRecoGeometryRcd>(edm::ESInputTag{});
    rdpToken_ = cc.consumesFrom<CSCRecoDigiParameters, CSCRecoDigiParametersRcd>(edm::ESInputTag{});
  }

  // Feed these value to where I need them
  applyAlignment_ = p.getParameter<bool>("applyAlignment");
  if (applyAlignment_) {
    globalPositionToken_ = cc.consumesFrom<Alignments, GlobalPositionRcd>(edm::ESInputTag{"", alignmentsLabel_});
    alignmentsToken_ = cc.consumesFrom<Alignments, CSCAlignmentRcd>(edm::ESInputTag{"", alignmentsLabel_});
    alignmentErrorsToken_ =
        cc.consumesFrom<AlignmentErrorsExtended, CSCAlignmentErrorExtendedRcd>(edm::ESInputTag{"", alignmentsLabel_});
  }

  edm::LogVerbatim("Geometry") << "@SUB=CSCGeometryESModule Label '" << myLabel_ << "' "
                               << (applyAlignment_ ? "looking for" : "IGNORING") << " alignment labels '"
                               << alignmentsLabel_ << "'.";
}

void CSCGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //
  // This cfi should be included to build the CSC geometry model.
  //
  // modelling flags (for completeness - internal defaults are already sane)
  // GF would like to have a shorter name (e.g. CSCGeometry), but since originally
  // there was no name, replace statements in other configs would not work anymore...
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("fromDD4hep", false);
  desc.add<std::string>("alignmentsLabel", "");
  desc.add<std::string>("appendToDataLabel", "");
  desc.add<bool>("useRealWireGeometry", true);
  desc.add<bool>("useOnlyWiresInME1a", false);
  desc.add<bool>("useGangedStripsInME1a", true);
  desc.add<bool>("useCentreTIOffsets", false);
  desc.add<bool>("applyAlignment", true);  //GF: to be abandoned
  desc.addUntracked<bool>("debugV", false);
  descriptions.add("CSCGeometryESModule", desc);
}

std::shared_ptr<CSCGeometry> CSCGeometryESModule::produce(const MuonGeometryRecord& record) {
  auto host = holder_.makeOrGet([this]() {
    return new HostType(
        debugV_, useGangedStripsInME1a_, useOnlyWiresInME1a_, useRealWireGeometry_, useCentreTIOffsets_);
  });

  initCSCGeometry_(record, host);

  // Called whenever the alignments or alignment errors change

  if (applyAlignment_) {
    // applyAlignment_ is scheduled for removal.
    // Ideal geometry obtained by using 'fake alignment' (with applyAlignment_ = true)
    const auto& globalPosition = record.get(globalPositionToken_);
    const auto& alignments = record.get(alignmentsToken_);
    const auto& alignmentErrors = record.get(alignmentErrorsToken_);
    // Only apply alignment if values exist
    if (alignments.empty() && alignmentErrors.empty() && globalPosition.empty()) {
      edm::LogVerbatim("Config") << "@SUB=CSCGeometryRecord::produce Alignment(Error)s and global position (label '"
                                 << alignmentsLabel_ << "') empty: Geometry producer (label "
                                 << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<CSCGeometry>(
          &(*host), &alignments, &alignmentErrors, align::DetectorGlobalPosition(globalPosition, DetId(DetId::Muon)));
    }
  }
  return host;  // automatically converts to std::shared_ptr<CSCGeometry>
}

void CSCGeometryESModule::initCSCGeometry_(const MuonGeometryRecord& record, std::shared_ptr<HostType>& host) {
  if (fromDDD_) {
    edm::LogVerbatim("CSCGeoemtryESModule") << "(0) CSCGeometryESModule  - DDD ";
    host->ifRecordChanges<IdealGeometryRecord>(record, [&host, &record, this](auto const& rec) {
      host->clear();
      edm::ESTransientHandle<DDCompactView> cpv = record.getTransientHandle(cpvToken_);
      const auto& mdc = rec.get(mdcToken_);
      CSCGeometryBuilderFromDDD builder;
      builder.build(*host, cpv.product(), mdc);
    });
  } else if (fromDD4hep_) {
    edm::LogVerbatim("CSCGeoemtryESModule") << "(0) CSCGeometryESModule  - DD4HEP ";
    host->ifRecordChanges<IdealGeometryRecord>(record, [&host, &record, this](auto const& rec) {
      host->clear();
      edm::ESTransientHandle<cms::DDCompactView> cpv = record.getTransientHandle(cpvTokendd4hep_);
      const auto& mdc = rec.get(mdcToken_);
      CSCGeometryBuilderFromDDD builder;
      builder.build(*host, cpv.product(), mdc);
    });
  } else {
    bool recreateGeometry = false;

    host->ifRecordChanges<CSCRecoGeometryRcd>(record,
                                              [&recreateGeometry](auto const& rec) { recreateGeometry = true; });

    host->ifRecordChanges<CSCRecoDigiParametersRcd>(record,
                                                    [&recreateGeometry](auto const& rec) { recreateGeometry = true; });
    edm::LogVerbatim("CSCGeoemtryESModule") << "(0) CSCGeometryESModule  - DB recreateGeometry=false ";
    if (recreateGeometry) {
      edm::LogVerbatim("CSCGeoemtryESModule") << "(0) CSCGeometryESModule  - DB recreateGeometry=true ";
      host->clear();
      const auto& rig = record.get(rigToken_);
      const auto& rdp = record.get(rdpToken_);
      CSCGeometryBuilder cscgb;
      cscgb.build(*host, rig, rdp);
    }
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(CSCGeometryESModule);
