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


class AlignmentProducer : public AlignmentProducerBase, public edm::ESProducerLooper
{

public:

  /// Constructor
  AlignmentProducer(const edm::ParameterSet&);

  /// Destructor
  ~AlignmentProducer() = default;

  /// Produce the tracker geometry
  virtual std::shared_ptr<TrackerGeometry> produceTracker(const TrackerDigiGeometryRecord&);

  /// Produce the muon DT geometry
  virtual std::shared_ptr<DTGeometry> produceDT(const MuonGeometryRecord&);

  /// Produce the muon CSC geometry
  virtual std::shared_ptr<CSCGeometry> produceCSC(const MuonGeometryRecord&);

  /// Called at beginning of job
  virtual void beginOfJob(const edm::EventSetup&) override;

  /// Called at end of job
  virtual void endOfJob() override;

  /// Called at beginning of loop
  virtual void startingNewLoop(unsigned int iLoop) override;

  /// Called at end of loop
  virtual Status endOfLoop(const edm::EventSetup&, unsigned int iLoop) override;

  /// Called at run start and calling algorithms beginRun
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /// Called at run end - currently reading TkFittedLasBeam if an InpuTag is given for that
  virtual void endRun(const edm::Run&, const edm::EventSetup&) override;

  /// Called at lumi block start, calling algorithm's beginLuminosityBlock
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&,
                                    const edm::EventSetup&) override;

  /// Called at lumi block end, calling algorithm's endLuminosityBlock
  virtual void endLuminosityBlock(const edm::LuminosityBlock&,
                                  const edm::EventSetup&) override;

  /// Called at each event
  virtual Status duringLoop(const edm::Event&, const edm::EventSetup&) override;

private:
  virtual bool getTrajTrackAssociationCollection(const edm::Event&,
                                                 edm::Handle<TrajTrackAssociationCollection>&) override;
  virtual bool getBeamSpot(const edm::Event&, edm::Handle<reco::BeamSpot>&) override;
  virtual bool getTkFittedLasBeamCollection(const edm::Run&,
                                            edm::Handle<TkFittedLasBeamCollection>&) override;
  virtual bool getTsosVectorCollection(const edm::Run&,
                                       edm::Handle<TsosVectorCollection>&) override;
  virtual bool getAliClusterValueMap(const edm::Event&,
                                     edm::Handle<AliClusterValueMap>&) override;

  const unsigned int maxLoops_;     /// Number of loops to loop

};


//------------------------------------------------------------------------------
inline
bool
AlignmentProducer::getTrajTrackAssociationCollection(const edm::Event& event,
                                                     edm::Handle<TrajTrackAssociationCollection>& result) {
  return event.getByLabel(tjTkAssociationMapTag_, result);
}


//------------------------------------------------------------------------------
inline
bool
AlignmentProducer::getBeamSpot(const edm::Event& event,
                               edm::Handle<reco::BeamSpot>& result) {
  return event.getByLabel(beamSpotTag_, result);
}


//------------------------------------------------------------------------------
inline
bool
AlignmentProducer::getTkFittedLasBeamCollection(const edm::Run& run,
                                                edm::Handle<TkFittedLasBeamCollection>& result) {
  return run.getByLabel(tkLasBeamTag_, result);
}


//------------------------------------------------------------------------------
inline
bool
AlignmentProducer::getTsosVectorCollection(const edm::Run& run,
                                           edm::Handle<TsosVectorCollection>& result) {
  return run.getByLabel(tkLasBeamTag_, result);
}


//------------------------------------------------------------------------------
inline
bool
AlignmentProducer::getAliClusterValueMap(const edm::Event& event,
                                         edm::Handle<AliClusterValueMap>& result) {
  return event.getByLabel(clusterValueMapTag_, result);
}

#endif
