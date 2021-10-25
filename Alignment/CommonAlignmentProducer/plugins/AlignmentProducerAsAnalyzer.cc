/**
 * @package   Alignment/CommonAlignmentProducer
 * @file      AlignmentProducerAsAnalyzer.cc
 *
 * @author    Max Stark (max.stark@cern.ch)
 * @date      2015/07/16
 */

/*** Header file ***/
#include "AlignmentProducerAsAnalyzer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//------------------------------------------------------------------------------
AlignmentProducerAsAnalyzer::AlignmentProducerAsAnalyzer(const edm::ParameterSet& config)
    : AlignmentProducerBase(config, consumesCollector()),
      token_(produces<AlignmentToken, edm::Transition::EndProcessBlock>()) {
  usesResource(TFileService::kSharedResource);

  tjTkAssociationMapToken_ = consumes<TrajTrackAssociationCollection>(tjTkAssociationMapTag_);
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpotTag_);
  tkLasBeamToken_ = consumes<TkFittedLasBeamCollection>(tkLasBeamTag_);
  tsosVectorToken_ = consumes<TsosVectorCollection>(tkLasBeamTag_);
  clusterValueMapToken_ = consumes<AliClusterValueMap>(clusterValueMapTag_);
}

//------------------------------------------------------------------------------
void AlignmentProducerAsAnalyzer::beginJob() {}

//------------------------------------------------------------------------------
void AlignmentProducerAsAnalyzer::endJob() {}

//------------------------------------------------------------------------------
void AlignmentProducerAsAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  beginRunImpl(run, setup);
}

//------------------------------------------------------------------------------
void AlignmentProducerAsAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& setup) { endRunImpl(run, setup); }

//------------------------------------------------------------------------------
void AlignmentProducerAsAnalyzer::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
                                                       const edm::EventSetup& setup) {
  beginLuminosityBlockImpl(lumiBlock, setup);
}

//------------------------------------------------------------------------------
void AlignmentProducerAsAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
                                                     const edm::EventSetup& setup) {
  endLuminosityBlockImpl(lumiBlock, setup);
}

void AlignmentProducerAsAnalyzer::endProcessBlockProduce(edm::ProcessBlock& processBlock) {
  const AlignmentToken valueToPut{};
  processBlock.emplace(token_, valueToPut);

  terminateProcessing();
  if (!finish()) {
    edm::LogError("Alignment") << "@SUB=AlignmentProducerAsAnalyzer::endJob"
                               << "Did not process any events, do not dare to store to DB.";
  }

  // message is used by the MillePede log parser to check the end of the job
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducerAsAnalyzer::endJob"
                            << "Finished alignment producer job.";
}

//------------------------------------------------------------------------------
void AlignmentProducerAsAnalyzer::accumulate(edm::Event const& event, edm::EventSetup const& setup) {
  processEvent(event, setup);
}

DEFINE_FWK_MODULE(AlignmentProducerAsAnalyzer);
