/// \file AlignmentProducer.cc
///
///  \author    : Frederic Ronga

#include "AlignmentProducer.h"

#include "FWCore/Framework/interface/LooperFactory.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//------------------------------------------------------------------------------
AlignmentProducer::AlignmentProducer(const edm::ParameterSet &config)
    : AlignmentProducerBase(config, consumesCollector()),
      maxLoops_{config.getUntrackedParameter<unsigned int>("maxLoops")} {
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::AlignmentProducer";

  // Tell the framework what data is being produced
  if (doTracker_) {
    setWhatProduced(this, &AlignmentProducer::produceTracker);
  }
}

//------------------------------------------------------------------------------
std::shared_ptr<TrackerGeometry> AlignmentProducer::produceTracker(const TrackerDigiGeometryRecord &) {
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::produceTracker";
  return trackerGeometry_;
}

//------------------------------------------------------------------------------
void AlignmentProducer::beginOfJob(const edm::EventSetup &iSetup) {
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob";
  initAlignmentAlgorithm(iSetup);
}

//------------------------------------------------------------------------------
void AlignmentProducer::endOfJob() {
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfJob";

  if (!finish()) {
    edm::LogError("Alignment") << "@SUB=AlignmentProducer::endOfJob"
                               << "Did not process any events in last loop, do not dare to store to DB.";
  }
}

//------------------------------------------------------------------------------
void AlignmentProducer::startingNewLoop(unsigned int iLoop) {
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::startingNewLoop"
                            << "Starting loop number " << iLoop;
  startProcessing();
}

//------------------------------------------------------------------------------
edm::EDLooper::Status AlignmentProducer::endOfLoop(const edm::EventSetup &iSetup, unsigned int iLoop) {
  if (0 == nEvent()) {
    // beginOfJob is usually called by the framework in the first event of the first loop
    // (a hack: beginOfJob needs the EventSetup that is not well defined without an event)
    // and the algorithms rely on the initialisations done in beginOfJob. We cannot call
    // this->beginOfJob(iSetup); here either since that will access the EventSetup to get
    // some geometry information that is not defined either without having seen an event.
    edm::LogError("Alignment") << "@SUB=AlignmentProducer::endOfLoop"
                               << "Did not process any events in loop " << iLoop
                               << ", stop processing without terminating algorithm.";
    return kStop;
  }

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfLoop"
                            << "Ending loop " << iLoop << ", terminating algorithm.";
  terminateProcessing(&iSetup);

  if (iLoop == maxLoops_ - 1 || iLoop >= maxLoops_)
    return kStop;
  else
    return kContinue;
}

//------------------------------------------------------------------------------
edm::EDLooper::Status AlignmentProducer::duringLoop(const edm::Event &event, const edm::EventSetup &setup) {
  if (processEvent(event, setup))
    return kContinue;
  else
    return kStop;
}

//------------------------------------------------------------------------------
void AlignmentProducer::beginRun(const edm::Run &run, const edm::EventSetup &setup) { beginRunImpl(run, setup); }

//------------------------------------------------------------------------------
void AlignmentProducer::endRun(const edm::Run &run, const edm::EventSetup &setup) { endRunImpl(run, setup); }

//------------------------------------------------------------------------------
void AlignmentProducer::beginLuminosityBlock(const edm::LuminosityBlock &lumiBlock, const edm::EventSetup &setup) {
  beginLuminosityBlockImpl(lumiBlock, setup);
}

//------------------------------------------------------------------------------
void AlignmentProducer::endLuminosityBlock(const edm::LuminosityBlock &lumiBlock, const edm::EventSetup &setup) {
  endLuminosityBlockImpl(lumiBlock, setup);
}

DEFINE_FWK_LOOPER(AlignmentProducer);
