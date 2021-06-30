#ifndef Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h
#define Alignment_CommonAlignmentAlgorithm_TrackerAlignmentProducer_h

/// \class AlignmentProducer
///
/// Package     : Alignment/CommonAlignmentProducer
/// Description : calls alignment algorithms
///
///  \author    : Frederic Ronga

#include "Alignment/CommonAlignmentProducer/interface/AlignmentProducerBase.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/Framework/interface/Run.h"

class AlignmentProducer : public edm::ESProducerLooper, public AlignmentProducerBase {
public:
  /// Constructor
  AlignmentProducer(const edm::ParameterSet&);

  /// Destructor
  ~AlignmentProducer() override = default;

  /// Produce the tracker geometry
  virtual std::shared_ptr<TrackerGeometry> produceTracker(const TrackerDigiGeometryRecord&);

  /// Called at beginning of job
  void beginOfJob(const edm::EventSetup&) override;

  /// Called at end of job
  void endOfJob() override;

  /// Called at beginning of loop
  void startingNewLoop(unsigned int iLoop) override;

  /// Called at end of loop
  Status endOfLoop(const edm::EventSetup&, unsigned int iLoop) override;

  /// Called at run start and calling algorithms beginRun
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /// Called at run end - currently reading TkFittedLasBeam if an InpuTag is given for that
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  /// Called at lumi block start, calling algorithm's beginLuminosityBlock
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

  /// Called at lumi block end, calling algorithm's endLuminosityBlock
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

  /// Called at each event
  Status duringLoop(const edm::Event&, const edm::EventSetup&) override;

private:
  bool getTrajTrackAssociationCollection(const edm::Event&, edm::Handle<TrajTrackAssociationCollection>&) override;
  bool getBeamSpot(const edm::Event&, edm::Handle<reco::BeamSpot>&) override;
  bool getTkFittedLasBeamCollection(const edm::Run&, edm::Handle<TkFittedLasBeamCollection>&) override;
  bool getTsosVectorCollection(const edm::Run&, edm::Handle<TsosVectorCollection>&) override;
  bool getAliClusterValueMap(const edm::Event&, edm::Handle<AliClusterValueMap>&) override;

  const unsigned int maxLoops_;  /// Number of loops to loop
};

//------------------------------------------------------------------------------
inline bool AlignmentProducer::getTrajTrackAssociationCollection(const edm::Event& event,
                                                                 edm::Handle<TrajTrackAssociationCollection>& result) {
  return event.getByLabel(tjTkAssociationMapTag_, result);
}

//------------------------------------------------------------------------------
inline bool AlignmentProducer::getBeamSpot(const edm::Event& event, edm::Handle<reco::BeamSpot>& result) {
  return event.getByLabel(beamSpotTag_, result);
}

//------------------------------------------------------------------------------
inline bool AlignmentProducer::getTkFittedLasBeamCollection(const edm::Run& run,
                                                            edm::Handle<TkFittedLasBeamCollection>& result) {
  return run.getByLabel(tkLasBeamTag_, result);
}

//------------------------------------------------------------------------------
inline bool AlignmentProducer::getTsosVectorCollection(const edm::Run& run, edm::Handle<TsosVectorCollection>& result) {
  return run.getByLabel(tkLasBeamTag_, result);
}

//------------------------------------------------------------------------------
inline bool AlignmentProducer::getAliClusterValueMap(const edm::Event& event, edm::Handle<AliClusterValueMap>& result) {
  return event.getByLabel(clusterValueMapTag_, result);
}

#endif
