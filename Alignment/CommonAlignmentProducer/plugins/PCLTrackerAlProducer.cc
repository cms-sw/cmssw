/**
 * @package   Alignment/CommonAlignmentProducer
 * @file      PCLTrackerAlProducer.cc
 *
 * @author    Max Stark (max.stark@cern.ch)
 * @date      2015/07/16
 */



/*** Header file ***/
#include "PCLTrackerAlProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"



//------------------------------------------------------------------------------
PCLTrackerAlProducer::PCLTrackerAlProducer(const edm::ParameterSet& config) :
  AlignmentProducerBase{config}
{
  tjTkAssociationMapToken_ = consumes<TrajTrackAssociationCollection>(tjTkAssociationMapTag_);
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpotTag_);
  tkLasBeamToken_ = consumes<TkFittedLasBeamCollection>(tkLasBeamTag_);
  tsosVectorToken_ = consumes<TsosVectorCollection>(tkLasBeamTag_);
  clusterValueMapToken_ = consumes<AliClusterValueMap>(clusterValueMapTag_);
}


//------------------------------------------------------------------------------
void
PCLTrackerAlProducer::beginJob()
{
}


//------------------------------------------------------------------------------
void
PCLTrackerAlProducer::endJob()
{
  terminateProcessing();
  if (!finish()) {
    edm::LogError("Alignment")
      << "@SUB=PCLTrackerAlProducer::endJob"
      << "Did not process any events, do not dare to store to DB.";
  }

  // message is used by the MillePede log parser to check the end of the job
  edm::LogInfo("Alignment")
    << "@SUB=PCLTrackerAlProducer::endJob"
    << "Finished alignment producer job.";
}


//------------------------------------------------------------------------------
void
PCLTrackerAlProducer::beginRun(const edm::Run& run, const edm::EventSetup& setup)
{
  beginRunImpl(run, setup);
}


//------------------------------------------------------------------------------
void
PCLTrackerAlProducer::endRun(const edm::Run& run, const edm::EventSetup& setup)
{
  endRunImpl(run, setup);
}


//------------------------------------------------------------------------------
void
PCLTrackerAlProducer::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
                                           const edm::EventSetup& setup)
{
  beginLuminosityBlockImpl(lumiBlock, setup);
}


//------------------------------------------------------------------------------
void
PCLTrackerAlProducer::endLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
                                         const edm::EventSetup& setup)
{
  endLuminosityBlockImpl(lumiBlock, setup);
}


//------------------------------------------------------------------------------
void
PCLTrackerAlProducer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  processEvent(event, setup);
}


DEFINE_FWK_MODULE(PCLTrackerAlProducer);
