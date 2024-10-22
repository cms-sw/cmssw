// -*- C++ -*-
//
// Package:    Alignment/TrackerAlignment
// Class:      CreateIdealTkAlRecords
//
/**\class CreateIdealTkAlRecords CreateIdealTkAlRecords.cc Alignment/TrackerAlignment/plugins/CreateIdealTkAlRecords.cc

 Description: Plugin to create ideal tracker alignment records.

 Implementation:
     The plugin takes the geometry stored in the global tag and transfers this
     information to the format needed in the TrackerAlignmentRcd. The APEs are
     set to zero for all det IDs of the tracker geometry and put into an
     TrackerAlignmentErrorExtendedRcd. In addition an empty
     TrackerSurfaceDeformationRcd is created corresponding to ideal surfaces.

     An option exists to align to the content of the used global tag. This is
     useful, if the geometry record and the tracker alignment records do not
     match.

*/
//
// Original Author:  Gregor Mittag
//         Created:  Tue, 26 Apr 2016 09:45:13 GMT
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "CondFormats/GeometryObjects/interface/PTrackerAdditionalParametersPerDet.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"

#include "CLHEP/Vector/RotationInterfaces.h"

//
// class declaration
//

class CreateIdealTkAlRecords : public edm::one::EDAnalyzer<> {
public:
  explicit CreateIdealTkAlRecords(const edm::ParameterSet&);
  ~CreateIdealTkAlRecords() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::string toString(const GeomDetEnumerators::SubDetector&);
  static GeomDetEnumerators::SubDetector toSubDetector(const std::string& sub);
  static std::vector<GeomDetEnumerators::SubDetector> toSubDetectors(const std::vector<std::string>& subs);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void clearAlignmentInfos();
  std::unique_ptr<TrackerGeometry> retrieveGeometry(const edm::EventSetup&);
  void addAlignmentInfo(const GeomDet&);
  void alignToGT(const edm::EventSetup&);
  void writeToDB();

  // ----------member data ---------------------------

  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;
  const edm::ESGetToken<PTrackerAdditionalParametersPerDet, PTrackerAdditionalParametersPerDetRcd> ptitpToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> aliToken_;
  const edm::ESGetToken<AlignmentErrorsExtended, TrackerAlignmentErrorExtendedRcd> aliErrorToken_;
  const edm::ESGetToken<AlignmentSurfaceDeformations, TrackerSurfaceDeformationRcd> aliSurfaceToken_;
  const std::vector<GeomDetEnumerators::SubDetector> skipSubDetectors_;
  const bool alignToGlobalTag_;
  const bool createReferenceRcd_;
  bool firstEvent_;
  Alignments alignments_;
  AlignmentErrorsExtended alignmentErrors_;
  AlignmentSurfaceDeformations alignmentSurfaceDeformations_;
  std::vector<uint32_t> rawIDs_;
  std::vector<GeomDetEnumerators::SubDetector> subDets_;
};

//
// constructors and destructor
//
CreateIdealTkAlRecords::CreateIdealTkAlRecords(const edm::ParameterSet& iConfig)
    : geomDetToken_(esConsumes()),
      ptpToken_(esConsumes()),
      ptitpToken_(esConsumes()),
      topoToken_(esConsumes()),
      aliToken_(esConsumes()),
      aliErrorToken_(esConsumes()),
      aliSurfaceToken_(esConsumes()),
      skipSubDetectors_(toSubDetectors(iConfig.getUntrackedParameter<std::vector<std::string> >("skipSubDetectors"))),
      alignToGlobalTag_(iConfig.getUntrackedParameter<bool>("alignToGlobalTag")),
      createReferenceRcd_(iConfig.getUntrackedParameter<bool>("createReferenceRcd")),
      firstEvent_(true) {}

CreateIdealTkAlRecords::~CreateIdealTkAlRecords() {}

//
// member functions
//

// ------------ method called for each event  ------------
void CreateIdealTkAlRecords::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  if (firstEvent_) {
    clearAlignmentInfos();
    const auto tracker = retrieveGeometry(iSetup);

    auto dets = tracker->dets();
    std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) {
      return a->geographicalId().rawId() < b->geographicalId().rawId();
    });

    for (const auto& det : dets)
      addAlignmentInfo(*det);
    if (alignToGlobalTag_ && !createReferenceRcd_)
      alignToGT(iSetup);
    writeToDB();
    firstEvent_ = false;
  }
}

std::string CreateIdealTkAlRecords::toString(const GeomDetEnumerators::SubDetector& sub) {
  switch (sub) {
    case GeomDetEnumerators::PixelBarrel:
      return "PixelBarrel";
    case GeomDetEnumerators::PixelEndcap:
      return "PixelEndcap";
    case GeomDetEnumerators::TIB:
      return "TIB";
    case GeomDetEnumerators::TOB:
      return "TOB";
    case GeomDetEnumerators::TID:
      return "TID";
    case GeomDetEnumerators::TEC:
      return "TEC";
    case GeomDetEnumerators::CSC:
      return "CSC";
    case GeomDetEnumerators::DT:
      return "DT";
    case GeomDetEnumerators::RPCBarrel:
      return "RPCBarrel";
    case GeomDetEnumerators::RPCEndcap:
      return "RPCEndcap";
    case GeomDetEnumerators::GEM:
      return "GEM";
    case GeomDetEnumerators::ME0:
      return "ME0";
    case GeomDetEnumerators::P2OTB:
      return "P2OTB";
    case GeomDetEnumerators::P2OTEC:
      return "P2OTEC";
    case GeomDetEnumerators::P1PXB:
      return "P1PXB";
    case GeomDetEnumerators::P1PXEC:
      return "P1PXEC";
    case GeomDetEnumerators::P2PXB:
      return "P2PXB";
    case GeomDetEnumerators::P2PXEC:
      return "P2PXEC";
    case GeomDetEnumerators::invalidDet:
      return "invalidDet";
    default:
      throw cms::Exception("UnknownSubdetector");
  }
}

GeomDetEnumerators::SubDetector CreateIdealTkAlRecords::toSubDetector(const std::string& sub) {
  if (sub == "PixelBarrel")
    return GeomDetEnumerators::PixelBarrel;
  else if (sub == "PixelEndcap")
    return GeomDetEnumerators::PixelEndcap;
  else if (sub == "TIB")
    return GeomDetEnumerators::TIB;
  else if (sub == "TOB")
    return GeomDetEnumerators::TOB;
  else if (sub == "TID")
    return GeomDetEnumerators::TID;
  else if (sub == "TEC")
    return GeomDetEnumerators::TEC;
  else if (sub == "CSC")
    return GeomDetEnumerators::CSC;
  else if (sub == "DT")
    return GeomDetEnumerators::DT;
  else if (sub == "RPCBarrel")
    return GeomDetEnumerators::RPCBarrel;
  else if (sub == "RPCEndcap")
    return GeomDetEnumerators::RPCEndcap;
  else if (sub == "GEM")
    return GeomDetEnumerators::GEM;
  else if (sub == "ME0")
    return GeomDetEnumerators::ME0;
  else if (sub == "P2OTB")
    return GeomDetEnumerators::P2OTB;
  else if (sub == "P2OTEC")
    return GeomDetEnumerators::P2OTEC;
  else if (sub == "P1PXB")
    return GeomDetEnumerators::P1PXB;
  else if (sub == "P1PXEC")
    return GeomDetEnumerators::P1PXEC;
  else if (sub == "P2PXB")
    return GeomDetEnumerators::P2PXB;
  else if (sub == "P2PXEC")
    return GeomDetEnumerators::P2PXEC;
  else if (sub == "invalidDet")
    return GeomDetEnumerators::invalidDet;
  else
    throw cms::Exception("UnknownSubdetector") << sub;
}

std::vector<GeomDetEnumerators::SubDetector> CreateIdealTkAlRecords::toSubDetectors(
    const std::vector<std::string>& subs) {
  std::vector<GeomDetEnumerators::SubDetector> result;
  result.reserve(subs.size());
  for (const auto& sub : subs)
    result.emplace_back(toSubDetector(sub));
  return result;
}

void CreateIdealTkAlRecords::clearAlignmentInfos() {
  alignments_.clear();
  alignmentErrors_.clear();
  alignmentSurfaceDeformations_ = AlignmentSurfaceDeformations{};
  rawIDs_.clear();
}

std::unique_ptr<TrackerGeometry> CreateIdealTkAlRecords::retrieveGeometry(const edm::EventSetup& iSetup) {
  const GeometricDet* geometricDet = &iSetup.getData(geomDetToken_);
  const PTrackerParameters& ptp = iSetup.getData(ptpToken_);
  const PTrackerAdditionalParametersPerDet* ptitp = &iSetup.getData(ptitpToken_);
  const TrackerTopology* tTopo = &iSetup.getData(topoToken_);

  TrackerGeomBuilderFromGeometricDet trackerBuilder;

  return std::unique_ptr<TrackerGeometry>{trackerBuilder.build(geometricDet, ptitp, ptp, tTopo)};
}

void CreateIdealTkAlRecords::addAlignmentInfo(const GeomDet& det) {
  const auto subDetector = toString(det.subDetector());
  const auto& detId = det.geographicalId().rawId();
  const auto& pos = det.position();
  const auto& rot = det.rotation();
  rawIDs_.push_back(detId);
  subDets_.push_back(det.subDetector());

  // TrackerAlignmentRcd entry
  if (createReferenceRcd_) {
    alignments_.m_align.emplace_back(AlignTransform(AlignTransform::Translation(), AlignTransform::Rotation(), detId));
  } else {
    const AlignTransform::Translation translation(pos.x(), pos.y(), pos.z());
    const AlignTransform::Rotation rotation(
        CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz()));
    const auto& eulerAngles = rotation.eulerAngles();
    LogDebug("Alignment") << "============================================================\n"
                          << "subdetector: " << subDetector << "\n"
                          << "detId:       " << detId << "\n"
                          << "------------------------------------------------------------\n"
                          << "     x: " << pos.x() << "\n"
                          << "     y: " << pos.y() << "\n"
                          << "     z: " << pos.z() << "\n"
                          << "   phi: " << eulerAngles.phi() << "\n"
                          << " theta: " << eulerAngles.theta() << "\n"
                          << "   psi: " << eulerAngles.psi() << "\n"
                          << "============================================================\n";
    alignments_.m_align.emplace_back(AlignTransform(translation, rotation, detId));
  }

  // TrackerAlignmentErrorExtendedRcd entry
  const AlignTransformError::SymMatrix zeroAPEs(6, 0);
  alignmentErrors_.m_alignError.emplace_back(AlignTransformErrorExtended(zeroAPEs, detId));
}

void CreateIdealTkAlRecords::alignToGT(const edm::EventSetup& iSetup) {
  LogDebug("Alignment") << "Aligning to global tag\n";

  const Alignments* alignments = &iSetup.getData(aliToken_);
  const AlignmentErrorsExtended* alignmentErrors = &iSetup.getData(aliErrorToken_);
  const AlignmentSurfaceDeformations* surfaceDeformations = &iSetup.getData(aliSurfaceToken_);

  if (alignments->m_align.size() != alignmentErrors->m_alignError.size())
    throw cms::Exception("GeometryMismatch")
        << "Size mismatch between alignments (size=" << alignments->m_align.size()
        << ") and alignment errors (size=" << alignmentErrors->m_alignError.size() << ")";

  std::vector<uint32_t> commonIDs;
  auto itAlignErr = alignmentErrors->m_alignError.cbegin();
  for (auto itAlign = alignments->m_align.cbegin(); itAlign != alignments->m_align.cend(); ++itAlign, ++itAlignErr) {
    const auto id = itAlign->rawId();
    auto found = std::find(rawIDs_.cbegin(), rawIDs_.cend(), id);
    if (found != rawIDs_.cend()) {
      if (id != itAlignErr->rawId())
        throw cms::Exception("GeometryMismatch") << "DetId mismatch between alignments (rawId=" << id
                                                 << ") and alignment errors (rawId=" << itAlignErr->rawId() << ")";

      const auto index = std::distance(rawIDs_.cbegin(), found);
      if (std::find(skipSubDetectors_.begin(), skipSubDetectors_.end(), subDets_[index]) != skipSubDetectors_.end())
        continue;

      if (alignments_.m_align[index].rawId() != alignmentErrors_.m_alignError[index].rawId())
        throw cms::Exception("GeometryMismatch")
            << "DetId mismatch between alignments (rawId=" << alignments_.m_align[index].rawId()
            << ") and alignment errors (rawId=" << alignmentErrors_.m_alignError[index].rawId() << ")";

      LogDebug("Alignment") << "============================================================\n"
                            << "\nGeometry content (" << toString(subDets_[index]) << ", "
                            << alignments_.m_align[index].rawId() << "):\n"
                            << "\tx: " << alignments_.m_align[index].translation().x()
                            << "\ty: " << alignments_.m_align[index].translation().y()
                            << "\tz: " << alignments_.m_align[index].translation().z()
                            << "\tphi: " << alignments_.m_align[index].rotation().phi()
                            << "\ttheta: " << alignments_.m_align[index].rotation().theta()
                            << "\tpsi: " << alignments_.m_align[index].rotation().psi()
                            << "============================================================\n";
      alignments_.m_align[index] = *itAlign;
      alignmentErrors_.m_alignError[index] = *itAlignErr;
      commonIDs.push_back(id);
      LogDebug("Alignment") << "============================================================\n"
                            << "Global tag content (" << toString(subDets_[index]) << ", "
                            << alignments_.m_align[index].rawId() << "):\n"
                            << "\tx: " << alignments_.m_align[index].translation().x()
                            << "\ty: " << alignments_.m_align[index].translation().y()
                            << "\tz: " << alignments_.m_align[index].translation().z()
                            << "\tphi: " << alignments_.m_align[index].rotation().phi()
                            << "\ttheta: " << alignments_.m_align[index].rotation().theta()
                            << "\tpsi: " << alignments_.m_align[index].rotation().psi()
                            << "============================================================\n";
    }
  }

  // - surface deformations are stored differently
  //   -> different treatment
  // - the above payloads contain also entries for ideal modules
  // - no entry is created for ideal surfaces
  //   -> size of surface deformation payload does not necessarily match the
  //      size of the other tracker alignment payload
  for (const auto& id : commonIDs) {
    // search for common raw ID in surface deformation items
    auto item = std::find_if(surfaceDeformations->items().cbegin(),
                             surfaceDeformations->items().cend(),
                             [&id](const auto& i) { return i.m_rawId == id; });
    if (item == surfaceDeformations->items().cend())
      continue;  // not found

    // copy surface deformation item
    const auto index = std::distance(surfaceDeformations->items().cbegin(), item);
    const auto beginEndPair = surfaceDeformations->parameters(index);
    std::vector<align::Scalar> params(beginEndPair.first, beginEndPair.second);
    alignmentSurfaceDeformations_.add(item->m_rawId, item->m_parametrizationType, params);
  }
}

void CreateIdealTkAlRecords::writeToDB() {
  const auto& since = cond::timeTypeSpecs[cond::runnumber].beginValue;

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (!poolDb.isAvailable()) {
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  }

  edm::LogInfo("Alignment") << "Writing ideal tracker-alignment records.";
  poolDb->writeOneIOV(alignments_, since, "TrackerAlignmentRcd");
  poolDb->writeOneIOV(alignmentErrors_, since, "TrackerAlignmentErrorExtendedRcd");
  poolDb->writeOneIOV(alignmentSurfaceDeformations_, since, "TrackerSurfaceDeformationRcd");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CreateIdealTkAlRecords::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment(
      "Creates ideal TrackerAlignmentRcd and TrackerAlignmentErrorExtendedRcd "
      "from the loaded tracker geometry. "
      "PoolDBOutputService must be set up for these records.");
  desc.addUntracked<bool>("alignToGlobalTag", false);
  desc.addUntracked<std::vector<std::string> >("skipSubDetectors", std::vector<std::string>{});
  desc.addUntracked<bool>("createReferenceRcd", false);
  descriptions.add("createIdealTkAlRecords", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CreateIdealTkAlRecords);
