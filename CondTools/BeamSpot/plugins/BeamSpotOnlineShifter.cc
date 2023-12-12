// -*- C++ -*-
//
// Package:    CondTools/BeamSpot
// Class:      BeamSpotOnlineShifter
//
/**\class BeamSpotOnlineShifter BeamSpotOnlineShifter.cc CondTools/BeamSpot/plugins/BeamSpotOnlineShifter.cc

 Description: EDAnalyzer to create a BeamSpotOnlineHLTObjectsRcd from a BeamSpotObjectsRcd (inserting some parameters manually)

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Sat, 06 May 2023 21:10:00 GMT
//
//

// system include files
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <ctime>

// user include files
#include "CondCore/AlignmentPlugins/interface/AlignmentPayloadInspectorHelper.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

//
// class declaration
//

class BeamSpotOnlineShifter : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit BeamSpotOnlineShifter(const edm::ParameterSet&);
  ~BeamSpotOnlineShifter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  cond::Time_t pack(uint32_t, uint32_t);
  template <class Record>
  void writeToDB(const edm::Event& iEvent,
                 const edm::EventSetup& iSetup,
                 const edm::ESGetToken<BeamSpotOnlineObjects, Record>& token);

private:
  const GlobalPoint getPixelBarycenter(const AlignmentPI::TkAlBarycenters barycenters, const bool isFullPixel);
  const GlobalPoint deltaAlignments(const Alignments* target,
                                    const Alignments* reference,
                                    const TrackerTopology& tTopo,
                                    const bool isFullPixel = false);

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd> hltToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> legacyToken_;

  edm::ESWatcher<BeamSpotOnlineHLTObjectsRcd> bsHLTWatcher_;
  edm::ESWatcher<BeamSpotOnlineLegacyObjectsRcd> bsLegayWatcher_;

  // IoV-structure
  GlobalPoint theShift_;
  const bool fIsHLT_;
  const bool fullPixel_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoTokenBR_;
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> refAliTokenBR_;
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> tarAliTokenBR_;
  const double xShift_;
  const double yShift_;
  const double zShift_;
  uint32_t fIOVStartRun_;
  uint32_t fIOVStartLumi_;
  cond::Time_t fnewSince_;
  bool fuseNewSince_;
  std::string fLabel_;
};

//
// constructors and destructor
//
BeamSpotOnlineShifter::BeamSpotOnlineShifter(const edm::ParameterSet& iConfig)
    : theShift_(GlobalPoint()),
      fIsHLT_(iConfig.getParameter<bool>("isHLT")),
      fullPixel_(iConfig.getParameter<bool>("useFullPixel")),
      trackerTopoTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      refAliTokenBR_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "reference"))),
      tarAliTokenBR_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "target"))),
      xShift_(iConfig.getParameter<double>("xShift")),
      yShift_(iConfig.getParameter<double>("yShift")),
      zShift_(iConfig.getParameter<double>("zShift")) {
  if (iConfig.exists("IOVStartRun") && iConfig.exists("IOVStartLumi")) {
    fIOVStartRun_ = iConfig.getUntrackedParameter<uint32_t>("IOVStartRun");
    fIOVStartLumi_ = iConfig.getUntrackedParameter<uint32_t>("IOVStartLumi");
    fnewSince_ = BeamSpotOnlineShifter::pack(fIOVStartRun_, fIOVStartLumi_);
    fuseNewSince_ = true;
    edm::LogPrint("BeamSpotOnlineShifter") << "useNewSince = True";
  } else {
    fuseNewSince_ = false;
    edm::LogPrint("BeamSpotOnlineShifter") << "useNewSince = False";
  }
  fLabel_ = (fIsHLT_) ? "BeamSpotOnlineHLTObjectsRcd" : "BeamSpotOnlineLegacyObjectsRcd";

  if (fIsHLT_) {
    hltToken_ = esConsumes();
  } else {
    legacyToken_ = esConsumes();
  }
}

//
// member functions
//

// ------------ Create a since object (cond::Time_t) by packing Run and LS (both uint32_t)  ------------
cond::Time_t BeamSpotOnlineShifter::pack(uint32_t fIOVStartRun, uint32_t fIOVStartLumi) {
  return ((uint64_t)fIOVStartRun << 32 | fIOVStartLumi);
}

template <class Record>
void BeamSpotOnlineShifter::writeToDB(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      const edm::ESGetToken<BeamSpotOnlineObjects, Record>& token) {
  // input object
  const BeamSpotOnlineObjects* inputSpot = &iSetup.getData(token);

  // output object
  BeamSpotOnlineObjects abeam;

  // N.B.: theShift is the difference between the target and the reference geometry barycenters
  // so if effectively the displacement of the new origin of reference frame w.r.t the old one.
  // This has to be subtracted from the old position of the beamspot:
  // - if the new reference frame rises, the beamspot drops
  // - if the new reference frame drops, the beamspot rises

  abeam.setPosition(inputSpot->x() - theShift_.x(), inputSpot->y() - theShift_.y(), inputSpot->z() - theShift_.z());
  abeam.setSigmaZ(inputSpot->sigmaZ());
  abeam.setdxdz(inputSpot->dxdz());
  abeam.setdydz(inputSpot->dydz());
  abeam.setBeamWidthX(inputSpot->beamWidthX());
  abeam.setBeamWidthY(inputSpot->beamWidthY());
  abeam.setBeamWidthXError(inputSpot->beamWidthXError());
  abeam.setBeamWidthYError(inputSpot->beamWidthYError());

  for (unsigned int i = 0; i < 7; i++) {
    for (unsigned int j = 0; j < 7; j++) {
      abeam.setCovariance(i, j, inputSpot->covariance(i, j));
    }
  }

  abeam.setType(inputSpot->beamType());
  abeam.setEmittanceX(inputSpot->emittanceX());
  abeam.setEmittanceY(inputSpot->emittanceY());
  abeam.setBetaStar(inputSpot->betaStar());

  // online BeamSpot object specific
  abeam.setLastAnalyzedLumi(inputSpot->lastAnalyzedLumi());
  abeam.setLastAnalyzedRun(inputSpot->lastAnalyzedRun());
  abeam.setLastAnalyzedFill(inputSpot->lastAnalyzedFill());
  abeam.setStartTimeStamp(inputSpot->startTimeStamp());
  abeam.setEndTimeStamp(inputSpot->endTimeStamp());
  abeam.setNumTracks(inputSpot->numTracks());
  abeam.setNumPVs(inputSpot->numPVs());
  abeam.setUsedEvents(inputSpot->usedEvents());
  abeam.setMaxPVs(inputSpot->maxPVs());
  abeam.setMeanPV(inputSpot->meanPV());
  abeam.setMeanErrorPV(inputSpot->meanErrorPV());
  abeam.setRmsPV(inputSpot->rmsPV());
  abeam.setRmsErrorPV(inputSpot->rmsErrorPV());
  abeam.setStartTime(inputSpot->startTime());
  abeam.setEndTime(inputSpot->endTime());
  abeam.setLumiRange(inputSpot->lumiRange());
  abeam.setCreationTime(inputSpot->creationTime());

  edm::LogPrint("BeamSpotOnlineShifter") << " Writing results to DB...";
  edm::LogPrint("BeamSpotOnlineShifter") << abeam;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    edm::LogPrint("BeamSpotOnlineShifter") << "poolDBService available";
    if (poolDbService->isNewTagRequest(fLabel_)) {
      edm::LogPrint("BeamSpotOnlineShifter") << "new tag requested";
      if (fuseNewSince_) {
        edm::LogPrint("BeamSpotOnlineShifter") << "Using a new Since: " << fnewSince_;
        poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, fnewSince_, fLabel_);
      } else
        poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->beginOfTime(), fLabel_);
    } else {
      edm::LogPrint("BeamSpotOnlineShifter") << "no new tag requested";
      if (fuseNewSince_) {
        cond::Time_t thisSince = BeamSpotOnlineShifter::pack(iEvent.getLuminosityBlock().run(),
                                                             iEvent.getLuminosityBlock().luminosityBlock());
        edm::LogPrint("BeamSpotOnlineShifter") << "Using a new Since: " << thisSince;
        poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, thisSince, fLabel_);
      } else
        poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->currentTime(), fLabel_);
    }
  }
  edm::LogPrint("BeamSpotOnlineShifter") << "[BeamSpotOnlineShifter] analyze done \n";
}

//_____________________________________________________________________________________________
const GlobalPoint BeamSpotOnlineShifter::deltaAlignments(const Alignments* target,
                                                         const Alignments* reference,
                                                         const TrackerTopology& tTopo,
                                                         const bool isFullPixel) {
  const std::map<AlignmentPI::coordinate, float> theZero = {
      {AlignmentPI::t_x, 0.0}, {AlignmentPI::t_y, 0.0}, {AlignmentPI::t_z, 0.0}};

  AlignmentPI::TkAlBarycenters ref_barycenters;
  ref_barycenters.computeBarycenters(reference->m_align, tTopo, theZero);
  const auto& ref = this->getPixelBarycenter(ref_barycenters, isFullPixel);

  AlignmentPI::TkAlBarycenters tar_barycenters;
  tar_barycenters.computeBarycenters(target->m_align, tTopo, theZero);
  const auto& tar = this->getPixelBarycenter(tar_barycenters, isFullPixel);

  return GlobalPoint(tar.x() - ref.x(), tar.y() - ref.y(), tar.z() - ref.z());
}

//_____________________________________________________________________________________________
const GlobalPoint BeamSpotOnlineShifter::getPixelBarycenter(AlignmentPI::TkAlBarycenters barycenters,
                                                            const bool isFullPixel) {
  const auto& BPix = barycenters.getPartitionAvg(AlignmentPI::PARTITION::BPIX);
  const double BPixMods = barycenters.getNModules(AlignmentPI::PARTITION::BPIX);

  const auto& FPixM = barycenters.getPartitionAvg(AlignmentPI::PARTITION::FPIXm);
  const double FPixMMods = barycenters.getNModules(AlignmentPI::PARTITION::FPIXm);

  const auto& FPixP = barycenters.getPartitionAvg(AlignmentPI::PARTITION::FPIXp);
  const double FPixPMods = barycenters.getNModules(AlignmentPI::PARTITION::FPIXp);

  const double BPixFrac = BPixMods / (BPixMods + FPixMMods + FPixPMods);
  const double FPixMFrac = FPixMMods / (BPixMods + FPixMMods + FPixPMods);
  const double FPixPFrac = FPixPMods / (BPixMods + FPixMMods + FPixPMods);

  if (isFullPixel) {
    return GlobalPoint(BPixFrac * BPix.x() + FPixMFrac * FPixM.x() + FPixPFrac * FPixP.x(),
                       BPixFrac * BPix.y() + FPixMFrac * FPixM.y() + FPixPFrac * FPixP.y(),
                       BPixFrac * BPix.z() + FPixMFrac * FPixM.z() + FPixPFrac * FPixP.z());
  } else {
    return GlobalPoint(BPix.x(), BPix.y(), BPix.z());
  }
}

//_____________________________________________________________________________________________
void BeamSpotOnlineShifter::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  const auto& reference = iSetup.getHandle(refAliTokenBR_);
  const auto& target = iSetup.getHandle(tarAliTokenBR_);

  const TrackerTopology& tTopo = iSetup.getData(trackerTopoTokenBR_);

  if (reference.isValid() and target.isValid()) {
    theShift_ = this->deltaAlignments(&(*reference), &(*target), tTopo, fullPixel_);
  } else {
    theShift_ = GlobalPoint(xShift_, yShift_, zShift_);
  }
  edm::LogPrint("BeamSpotOnlineShifter") << "[BeamSpotOnlineShifter] applied shift: " << theShift_ << std::endl;
}

// ------------ method called for each event  ------------
void BeamSpotOnlineShifter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  if (fIsHLT_) {
    if (bsHLTWatcher_.check(iSetup)) {
      writeToDB<BeamSpotOnlineHLTObjectsRcd>(iEvent, iSetup, hltToken_);
    }
  } else {
    if (bsLegayWatcher_.check(iSetup)) {
      writeToDB<BeamSpotOnlineLegacyObjectsRcd>(iEvent, iSetup, legacyToken_);
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamSpotOnlineShifter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("isHLT", true);
  desc.add<bool>("useFullPixel", false)->setComment("use the full pixel detector to compute the barycenter");
  desc.add<double>("xShift", 0.0)->setComment("in cm");
  desc.add<double>("yShift", 0.0)->setComment("in cm");
  desc.add<double>("zShift", 0.0)->setComment("in cm");
  desc.addOptionalUntracked<uint32_t>("IOVStartRun", 1);
  desc.addOptionalUntracked<uint32_t>("IOVStartLumi", 1);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlineShifter);
