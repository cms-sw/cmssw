#ifndef Alignment_CommonAlignmentProducer_PCLTrackerAlProducer_h
#define Alignment_CommonAlignmentProducer_PCLTrackerAlProducer_h

/**
 * @package   Alignment/CommonAlignmentProducer
 * @file      PCLTrackerAlProducer.h
 *
 * @author    Max Stark (max.stark@cern.ch)
 * @date      2015/07/16
 *
 * @brief     Tracker-AlignmentProducer for Prompt Calibration Loop (PCL)
 *
 * Code is based on standard offline AlignmentProducer (see AlignmentProducer.h)
 * Main difference is the base-class exchange from an ESProducerLooper to an
 * EDAnalyzer. For further information regarding aligment workflow on PCL see:
 *
 * https://indico.cern.ch/event/394130/session/0/contribution/8/attachments/1127471/1610233/2015-07-16_PixelPCL_Ali.pdf
 *
 */


#include "Alignment/CommonAlignmentProducer/interface/AlignmentProducerBase.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"


class PCLTrackerAlProducer : public AlignmentProducerBase, public edm::EDAnalyzer
{
  //========================== PUBLIC METHODS ==================================
public: //====================================================================

  /// Constructor
  PCLTrackerAlProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~PCLTrackerAlProducer() = default;

  /*** Code which implements the interface
       Called from outside ***/

  virtual void beginJob() override;
  virtual void endJob() override;

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&, const edm::EventSetup&) override;

  virtual void beginLuminosityBlock(const edm::LuminosityBlock&,
                                    const edm::EventSetup&) override;
  virtual void endLuminosityBlock(const edm::LuminosityBlock&,
                                  const edm::EventSetup&) override;

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

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

  edm::EDGetTokenT<TrajTrackAssociationCollection> tjTkAssociationMapToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<TkFittedLasBeamCollection> tkLasBeamToken_;
  edm::EDGetTokenT<TsosVectorCollection> tsosVectorToken_;
  edm::EDGetTokenT<AliClusterValueMap> clusterValueMapToken_;

};


//------------------------------------------------------------------------------
inline
bool
PCLTrackerAlProducer::getTrajTrackAssociationCollection(const edm::Event& event,
                                                        edm::Handle<TrajTrackAssociationCollection>& result) {
  return event.getByToken(tjTkAssociationMapToken_, result);
}


//------------------------------------------------------------------------------
inline
bool
PCLTrackerAlProducer::getBeamSpot(const edm::Event& event,
                                  edm::Handle<reco::BeamSpot>& result) {
  return event.getByToken(beamSpotToken_, result);
}


//------------------------------------------------------------------------------
inline
bool
PCLTrackerAlProducer::getTkFittedLasBeamCollection(const edm::Run& run,
                                                   edm::Handle<TkFittedLasBeamCollection>& result) {
  return run.getByToken(tkLasBeamToken_, result);
}


//------------------------------------------------------------------------------
inline
bool
PCLTrackerAlProducer::getTsosVectorCollection(const edm::Run& run,
                                              edm::Handle<TsosVectorCollection>& result) {
  return run.getByToken(tsosVectorToken_, result);
}


//------------------------------------------------------------------------------
inline
bool
PCLTrackerAlProducer::getAliClusterValueMap(const edm::Event& event,
                                            edm::Handle<AliClusterValueMap>& result) {
  return event.getByToken(clusterValueMapToken_, result);
}

#endif
