#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeomBuilderFromGeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

// Alignments
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDSurfaceDeformationRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>
#include <string>

class MTDDigiGeometryESModule : public edm::ESProducer {
public:
  MTDDigiGeometryESModule(const edm::ParameterSet& p);
  std::unique_ptr<MTDGeometry> produce(const MTDDigiGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /// Called when geometry description changes
  const std::string alignmentsLabel_;
  const std::string myLabel_;

  edm::ESGetToken<GeometricTimingDet, IdealGeometryRecord> geomTimingDetToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdTopoToken_;
  edm::ESGetToken<PMTDParameters, PMTDParametersRcd> pmtdParamsToken_;

  //alignment
  edm::ESGetToken<Alignments, GlobalPositionRcd> globalAlignToken_;
  edm::ESGetToken<Alignments, MTDAlignmentRcd> mtdAlignToken_;
  edm::ESGetToken<AlignmentErrorsExtended, MTDAlignmentErrorExtendedRcd> alignErrorsToken_;
  edm::ESGetToken<AlignmentSurfaceDeformations, MTDSurfaceDeformationRcd> deformationsToken_;

  const bool applyAlignment_;  // Switch to apply alignment corrections
  const bool fromDDD_;
};

//__________________________________________________________________
MTDDigiGeometryESModule::MTDDigiGeometryESModule(const edm::ParameterSet& p)
    : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
      myLabel_(p.getParameter<std::string>("appendToDataLabel")),
      applyAlignment_(p.getParameter<bool>("applyAlignment")),
      fromDDD_(p.getParameter<bool>("fromDDD"))

{
  auto cc = setWhatProduced(this);
  const edm::ESInputTag kEmpty;
  geomTimingDetToken_ = cc.consumesFrom<GeometricTimingDet, IdealGeometryRecord>(kEmpty);
  mtdTopoToken_ = cc.consumesFrom<MTDTopology, MTDTopologyRcd>(kEmpty);
  pmtdParamsToken_ = cc.consumesFrom<PMTDParameters, PMTDParametersRcd>(kEmpty);

  {
    const edm::ESInputTag kAlignTag{"", alignmentsLabel_};
    globalAlignToken_ = cc.consumesFrom<Alignments, GlobalPositionRcd>(kAlignTag);
    mtdAlignToken_ = cc.consumesFrom<Alignments, MTDAlignmentRcd>(kAlignTag);
    alignErrorsToken_ = cc.consumesFrom<AlignmentErrorsExtended, MTDAlignmentErrorExtendedRcd>(kAlignTag);
    deformationsToken_ = cc.consumesFrom<AlignmentSurfaceDeformations, MTDSurfaceDeformationRcd>(kAlignTag);
  }

  edm::LogInfo("Geometry") << "@SUB=MTDDigiGeometryESModule"
                           << "Label '" << myLabel_ << "' " << (applyAlignment_ ? "looking for" : "IGNORING")
                           << " alignment labels '" << alignmentsLabel_ << "'.";
}

//__________________________________________________________________
void MTDDigiGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription descDB;
  descDB.add<std::string>("appendToDataLabel", "");
  descDB.add<bool>("fromDDD", false);
  descDB.add<bool>("applyAlignment", true);
  descDB.add<std::string>("alignmentsLabel", "");
  descriptions.add("mtdGeometryDB", descDB);

  edm::ParameterSetDescription desc;
  desc.add<std::string>("appendToDataLabel", "");
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("applyAlignment", true);
  desc.add<std::string>("alignmentsLabel", "");
  descriptions.add("mtdGeometry", desc);
}

//__________________________________________________________________
std::unique_ptr<MTDGeometry> MTDDigiGeometryESModule::produce(const MTDDigiGeometryRecord& iRecord) {
  //
  // Called whenever the alignments, alignment errors or global positions change
  //
  GeometricTimingDet const& gD = iRecord.get(geomTimingDetToken_);

  MTDTopology const& tTopo = iRecord.get(mtdTopoToken_);

  PMTDParameters const& ptp = iRecord.get(pmtdParamsToken_);

  MTDGeomBuilderFromGeometricTimingDet builder;
  std::unique_ptr<MTDGeometry> mtd(builder.build(&(gD), ptp, &tTopo));

  if (applyAlignment_) {
    // Since fake is fully working when checking for 'empty', we should get rid of applyAlignment_!
    Alignments const& globalPosition = iRecord.get(globalAlignToken_);
    Alignments const& alignments = iRecord.get(mtdAlignToken_);
    AlignmentErrorsExtended const& alignmentErrors = iRecord.get(alignErrorsToken_);
    // apply if not empty:
    if (alignments.empty() && alignmentErrors.empty() && globalPosition.empty()) {
      edm::LogInfo("Config") << "@SUB=MTDDigiGeometryRecord::produce"
                             << "Alignment(Error)s and global position (label '" << alignmentsLabel_
                             << "') empty: Geometry producer (label "
                             << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.applyAlignments<MTDGeometry>(mtd.get(),
                                       &(alignments),
                                       &(alignmentErrors),
                                       align::DetectorGlobalPosition(globalPosition, DetId(DetId::Forward)));
    }

    AlignmentSurfaceDeformations const& surfaceDeformations = iRecord.get(deformationsToken_);
    // apply if not empty:
    if (surfaceDeformations.empty()) {
      edm::LogInfo("Config") << "@SUB=MTDDigiGeometryRecord::produce"
                             << "AlignmentSurfaceDeformations (label '" << alignmentsLabel_
                             << "') empty: Geometry producer (label "
                             << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.attachSurfaceDeformations<MTDGeometry>(mtd.get(), &(surfaceDeformations));
    }
  }

  return mtd;
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDDigiGeometryESModule);
