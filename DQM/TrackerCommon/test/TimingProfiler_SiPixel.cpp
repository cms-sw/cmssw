#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <cmath>

int main(int argc, char **argv) {

  //std::string file = "TimingInfo.txt";
  std::string file;
  file.assign(argv[1]);
  std::map<std::string,double> timingPerModule, timingPerLabel;
  std::map<unsigned,double> timingPerEvent;
  std::ifstream myTimingFile(file.c_str(),std::ifstream::in);
  std::string dummy1, label, module;
  double timing;
  unsigned idummy1,evt;
  int nbofevts = 0;

  // If the machine is busy, the factor is not 100%.
  double factor = 0.995;

  if ( myTimingFile ) {
    while ( !myTimingFile.eof() ) { 
      myTimingFile >> dummy1 >> evt >> idummy1 >> label >> module >> timing ;
      // std::cout << evt << " " << module << " " << timing << std::endl;
      timingPerEvent[evt]              += timing * factor * 1000.;	
      timingPerModule[module]          += timing * factor * 1000.;
      timingPerLabel[module+":"+label] += timing * factor * 1000.;
    }
    nbofevts = (int) timingPerEvent.size();
  } else {
    std::cout << "File " << file << " does not exist!" << std::endl;
  }

  std::map<std::string,double>::const_iterator modIt = timingPerModule.begin();
  std::map<std::string,double>::const_iterator labIt = timingPerLabel.begin();
  std::map<std::string,double>::const_iterator modEnd = timingPerModule.end();
  std::map<std::string,double>::const_iterator labEnd = timingPerLabel.end();
  std::map<double,std::string> modulePerTiming;
  std::map<double,std::string> labelPerTiming;

  for ( ; modIt != modEnd; ++modIt ) {
    double time = modIt->second/((double)nbofevts-1.);
    std::string name = modIt->first;
    modulePerTiming[time] = name;
  }
    
  for ( ; labIt != labEnd; ++labIt ) {
    double time = labIt->second/((double)nbofevts-1.);
    std::string name = labIt->first;
    labelPerTiming[time] = name;
  }
    
  std::map<double,std::string>::const_reverse_iterator timeIt = modulePerTiming.rbegin();
  std::map<double,std::string>::const_reverse_iterator timeEnd = modulePerTiming.rend();
  std::map<double,std::string>::const_reverse_iterator timeIt2 = labelPerTiming.rbegin();
  std::map<double,std::string>::const_reverse_iterator timeEnd2 = labelPerTiming.rend();

  std::cout << "Timing per module " << std::endl;
  std::cout << "================= " << std::endl;
  double totalTime = 0.;
  unsigned i=1;
  for ( ; timeIt != timeEnd; ++timeIt ) {

    totalTime += timeIt->first;
    std::cout << std::setw(3) << i++ 
	      << std::setw(50) << timeIt->second << " : " 
	      << std::setw(7) << std::setprecision(3) << timeIt-> first << " ms/event"
	      << std::endl;
  }
  std::cout << "Total time = " << totalTime << " ms/event " << std::endl;

/*
  std::cout << "================= " << std::endl;
  std::cout << "Timing per label  " << std::endl;
  std::cout << "================= " << std::endl;
  totalTime = 0.;
  i = 1;
  for ( ; timeIt2 != timeEnd2; ++timeIt2 ) {

    totalTime += timeIt2->first;
    std::cout << std::setw(3) << i++ 
	      << std::setw(100) << timeIt2->second << " : " 
	      << std::setw(7) << std::setprecision(3) << timeIt2-> first << " ms/event"
	      << std::endl;
  }
*/

  double subtotaltimepermodule = 0;
  double cumulativetimepermodule = 0;
  
  std::cout << "================= " << std::endl;
  std::cout << " DQM for collision : Timing per step " << std::endl;
  std::cout << "================= " << std::endl;

  std::cout << "1. Reconstruction " << std::endl;

  std::cout << "	1.1 Raw2Digi+LocalReco : " << std::endl;
  std::cout << "		1.1.1 : Raw2Digi " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SiPixelRawToDigi" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiPixelRawToDigi"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiPixelRawToDigi"];
  cumulativetimepermodule += timingPerModule["SiPixelRawToDigi"];
  std::cout << "		- " << std::setw(40) << "SiStripRawToDigi" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripRawToDigiModule"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiStripRawToDigiModule"];
  cumulativetimepermodule += timingPerModule["SiStripRawToDigiModule"];
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		1.1.2 : LocalReco" << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SiPixelClusterProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiPixelClusterProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiPixelClusterProducer"];
  cumulativetimepermodule += timingPerModule["SiPixelClusterProducer"];
  std::cout << "		- " << std::setw(40) << "SiPixelRecHitConverter" << std::setw(30) << "" << std::setw(8) << timingPerLabel["SiPixelRecHitConverter:siPixelRecHits"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerLabel["SiPixelRecHitConverter:siPixelRecHits"];
  cumulativetimepermodule += timingPerLabel["SiPixelRecHitConverter:siPixelRecHits"];
  std::cout << "		- " << std::setw(40) << "SiStripZeroSuppression" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripZeroSuppression"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiStripZeroSuppression"];
  cumulativetimepermodule += timingPerModule["SiStripZeroSuppression"];
  std::cout << "		- " << std::setw(40) << "SiStripClusterizer"     << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripClusterizer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiStripClusterizer"];
  cumulativetimepermodule += timingPerModule["SiStripClusterizer"];
  std::cout << "		- " << std::setw(40) << "SiStripRecHitConverter" << std::setw(30) << "" << std::setw(8) << timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHits"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHits"];
  cumulativetimepermodule += timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHits"];
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;


  std::cout << "	1.2 BeamSpot+RecoPixelVertexing+CkfTracks :" << std::endl;
  std::cout << "		1.2.1 : BeamSpot " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "BeamSpotProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["BeamSpotProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["BeamSpotProducer"];
  cumulativetimepermodule += timingPerModule["BeamSpotProducer"];
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		1.2.2 : RecoPixelVertexing " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "PixelTrackProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["PixelTrackProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["PixelTrackProducer"];
  cumulativetimepermodule += timingPerModule["PixelTrackProducer"];
  std::cout << "		- " << std::setw(40) << "PixelVertexProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["PixelVertexProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["PixelVertexProducer"];
  cumulativetimepermodule += timingPerModule["PixelVertexProducer"];
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		1.2.3 : CkfTracks " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SeedGeneratorFromRegionHitsEDProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["SeedGeneratorFromRegionHitsEDProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SeedGeneratorFromRegionHitsEDProducer"];
  cumulativetimepermodule += timingPerModule["SeedGeneratorFromRegionHitsEDProducer"];
/*
			- newSeedFromTriplets
			- newSeedFromPairs
			- secTriplets
			- thPLSeeds
			- fourthPLSeeds
			- fifthSeeds
*/
  std::cout << "		- " << std::setw(40) << "CkfTrackCandidateMaker" << std::setw(30) << "" << std::setw(8) << timingPerModule["CkfTrackCandidateMaker"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["CkfTrackCandidateMaker"];
  cumulativetimepermodule += timingPerModule["CkfTrackCandidateMaker"];
/*
			- newTrackCandidateMaker
			- secTrackCandidates
			- thTrackCandidates
			- fourthTrackCandidates
			- fifthTrackCandidates
*/
  std::cout << "		- " << std::setw(40) << "TrackProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["TrackProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["TrackProducer"];
  cumulativetimepermodule += timingPerModule["TrackProducer"];
/*
			- preFilterZeroStepTracks
			- preFilterStepOneTracks
			- secWithMaterialTracks
			- thWithMaterialTracks
			- fourthWithMaterialTracks
			- fifthWithMaterialTracks
*/
  std::cout << "		- " << std::setw(40) << "AnalyticalTrackSelector" << std::setw(30) << "" << std::setw(8) << timingPerModule["AnalyticalTrackSelector"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["AnalyticalTrackSelector"];
  cumulativetimepermodule += timingPerModule["AnalyticalTrackSelector"];
/*
			- zeroStepWithLooseQuality
			- zeroStepWithTightQuality
			- zeroStepTracksWithQuality
			- firstStepWithLooseQuality
			- firstStepWithTightQuality
			- preMergingFirstStepTracksWithQuality
			- secStepVtxLoose
			- secStepTrkLoose
			- secStepVtxTight
			- secStepTrkTight
			- secStepVtx
			- secStepTrk
			- thStepVtxLoose
			- thStepTrkLoose
			- thStepVtxTight
			- thStepTrkTight
			- thStepVtx
			- thStepTrk
			- pixellessStepLoose
			- pixellessStepTight
			- pixellessStep
			- tobtecStepLoose
			- tobtecStepTight
			- tobtecStep
*/
  std::cout << "		- " << std::setw(40) << "QualityFilter" << std::setw(30) << "" << std::setw(8) << timingPerModule["QualityFilter"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["QualityFilter"];
  cumulativetimepermodule += timingPerModule["QualityFilter"];
/*
			- zeroStepFilter
			- firstfilter
			- secfilter
			- thfilter
			- fourthfilter
*/
  std::cout << "		- " << std::setw(40) << "TrackClusterRemover" << std::setw(30) << "" << std::setw(8) << timingPerModule["TrackClusterRemover"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["TrackClusterRemover"];
  cumulativetimepermodule += timingPerModule["TrackClusterRemover"];
/*
			- newClusters
			- secClusters
			- thClusters
			- fourthClusters
			- fifthClusters
*/
  std::cout << "		- " << std::setw(40) << "SiPixelRecHitConverter" << std::setw(30) << "" << std::setw(8) << timingPerLabel["SiPixelRecHitConverter:newPixelRecHits"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiPixelRecHitConverter:secPixelRecHits"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiPixelRecHitConverter:thPixelRecHits"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiPixelRecHitConverter:fourthPixelRecHits"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiPixelRecHitConverter:fifthPixelRecHits"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerLabel["SiPixelRecHitConverter:newPixelRecHits"];
  subtotaltimepermodule   += timingPerLabel["SiPixelRecHitConverter:secPixelRecHits"];
  subtotaltimepermodule   += timingPerLabel["SiPixelRecHitConverter:thPixelRecHits"];
  subtotaltimepermodule   += timingPerLabel["SiPixelRecHitConverter:fourthPixelRecHits"];
  subtotaltimepermodule   += timingPerLabel["SiPixelRecHitConverter:fifthPixelRecHits"];
  cumulativetimepermodule += timingPerLabel["SiPixelRecHitConverter:newPixelRecHits"];
  cumulativetimepermodule += timingPerLabel["SiPixelRecHitConverter:secPixelRecHits"];
  cumulativetimepermodule += timingPerLabel["SiPixelRecHitConverter:thPixelRecHits"];
  cumulativetimepermodule += timingPerLabel["SiPixelRecHitConverter:fourthPixelRecHits"];
  cumulativetimepermodule += timingPerLabel["SiPixelRecHitConverter:fifthPixelRecHits"];
/*
			- newPixelRecHits
			- secPixelRecHits
			- thPixelRecHits
			- fourthPixelRecHits
			- fifthPixelRecHits
*/
  std::cout << "		- " << std::setw(40) << "SiStripRecHitConverter" << std::setw(30) << "" << std::setw(8) << timingPerLabel["SiStripRecHitConverter:newStripRecHits"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiStripRecHitConverter:secStripRecHits"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiStripRecHitConverter:thStripRecHits"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiStripRecHitConverter:fourthStripRecHits"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiStripRecHitConverter:fifthStripRecHits"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerLabel["SiStripRecHitConverter:newStripRecHits"];
  subtotaltimepermodule   += timingPerLabel["SiStripRecHitConverter:secStripRecHits"];
  subtotaltimepermodule   += timingPerLabel["SiStripRecHitConverter:thStripRecHits"];
  subtotaltimepermodule   += timingPerLabel["SiStripRecHitConverter:fourthStripRecHits"];
  subtotaltimepermodule   += timingPerLabel["SiStripRecHitConverter:fifthStripRecHits"];
  cumulativetimepermodule += timingPerLabel["SiStripRecHitConverter:newStripRecHits"];
  cumulativetimepermodule += timingPerLabel["SiStripRecHitConverter:secStripRecHits"];
  cumulativetimepermodule += timingPerLabel["SiStripRecHitConverter:thStripRecHits"];
  cumulativetimepermodule += timingPerLabel["SiStripRecHitConverter:fourthStripRecHits"];
  cumulativetimepermodule += timingPerLabel["SiStripRecHitConverter:fifthStripRecHits"];
/*
			- newStripRecHits
			- secStripRecHits
			- thStripRecHits
			- fourthStripRecHits
			- fifthStripRecHits
*/
  std::cout << "		- " << std::setw(40) << "SimpleTrackListMerger" << std::setw(30) << "" << std::setw(8) << timingPerModule["SimpleTrackListMerger"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SimpleTrackListMerger"];
  cumulativetimepermodule += timingPerModule["SimpleTrackListMerger"];
/*
			- merge2nd3rdTracks
			- merge4th5thTracks
			- iterTracks
			- generalTracks
*/
  std::cout << "		- " << std::setw(40) << "SeedCombiner" << std::setw(30) << "" << std::setw(8) << timingPerModule["SeedCombiner"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SeedCombiner"];
  cumulativetimepermodule += timingPerModule["SeedCombiner"];
/*
			- newCombinedSeeds
*/
  std::cout << "		- " << std::setw(40) << "DeDxEstimatorProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["DeDxEstimatorProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["DeDxEstimatorProducer"];
  cumulativetimepermodule += timingPerModule["DeDxEstimatorProducer"];
/*
			- dedxMedian
			- dedxHarmonic2
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;


  std::cout << "2. Data quality monitoring : " << std::endl;

  std::cout << "	2.1 DQM common modules " << std::endl;
  std::cout << "		2.1.1 : Quality tests " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "QualityTester" << std::setw(30) << "" << std::setw(8) << timingPerModule["QualityTester"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["QualityTester"];
  cumulativetimepermodule += timingPerModule["QualityTester"];
/*
			- qTester
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		2.1.2 : DQM playback environment " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "DQMEventInfo" << std::setw(30) << "" << std::setw(8) << timingPerLabel["DQMEventInfo:dqmEnv"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerLabel["DQMEventInfo:dqmEnv"];
  cumulativetimepermodule += timingPerLabel["DQMEventInfo:dqmEnv"];
/*
			- dqmEnv
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
//   std::cout << "		2.1.3 : DQM playback for Tracking info " << std::endl;
//   subtotaltimepermodule  = 0;
//   std::cout << "		- " << std::setw(40) << "DQMEventInfo" << std::setw(30) << "" << std::setw(8) << timingPerLabel["DQMEventInfo:dqmEnvTr"]/((double)nbofevts-1.) << " ms/event" << std::endl;
//   subtotaltimepermodule   += timingPerLabel["DQMEventInfo:dqmEnvTr"];
//   cumulativetimepermodule += timingPerLabel["DQMEventInfo:dqmEnvTr"];
// /*
// 			- dqmEnvTr
// */
//   std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
//   std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		2.1.3 : DQM file saver " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "DQMFileSaver" << std::setw(30) << "" << std::setw(8) << timingPerModule["DQMFileSaver"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["DQMFileSaver"];
  cumulativetimepermodule += timingPerModule["DQMFileSaver"];
/*
			- dqmSaver
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;

  std::cout << "	2.2 DQM monitoring " << std::endl;
  std::cout << "		2.2.1 :  Raw data error monitor" << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SiPixelRawDataErrorSource" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiPixelRawDataErrorSource"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiPixelRawDataErrorSource"];
  cumulativetimepermodule += timingPerModule["SiPixelRawDataErrorSource"];
/*
			- SiPixelRawDataErrorSource
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;

//   std::cout << "		- " << std::setw(40) << "SiStripMonitorCluster" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripMonitorCluster"]/((double)nbofevts-1.) << " ms/event" << std::endl;
//   subtotaltimepermodule   += timingPerModule["SiStripMonitorCluster"];
//   cumulativetimepermodule += timingPerModule["SiStripMonitorCluster"];
// /*
// 			- SiStripMonitorClusterReal
// */
//   std::cout << "		- " << std::setw(40) << "SiStripMonitorTrack" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripMonitorTrack"]/((double)nbofevts-1.) << " ms/event" << std::endl;
//   subtotaltimepermodule   += timingPerModule["SiStripMonitorTrack"];
//   cumulativetimepermodule += timingPerModule["SiStripMonitorTrack"];
// /*
// 			- SiStripMonitorTrack_gentk
// */
//   std::cout << "		- " << std::setw(40) << "MonitorTrackResiduals" << std::setw(30) << "" << std::setw(8) << timingPerModule["MonitorTrackResiduals"]/((double)nbofevts-1.) << " ms/event" << std::endl;
//   subtotaltimepermodule   += timingPerModule["MonitorTrackResiduals"];
//   cumulativetimepermodule += timingPerModule["MonitorTrackResiduals"];
// /*
// 			- MonitorTrackResiduals_gentk
// */
//   std::cout << "		- " << std::setw(40) << "TrackingMonitor" << std::setw(30) << "" << std::setw(8) << timingPerModule["TrackingMonitor"]/((double)nbofevts-1.) << " ms/event" << std::endl;
//   subtotaltimepermodule   += timingPerModule["TrackingMonitor"];
//   cumulativetimepermodule += timingPerModule["TrackingMonitor"];
// /*
// 			- TrackMon_gentk
// */
//   std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
//   std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		2.2.2 : Digi/Cluster/RecHit monitor " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SiPixelDigiSource" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiPixelDigiSource"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiPixelDigiSource"];
  cumulativetimepermodule += timingPerModule["SiPixelDigiSource"];
/*
			- SiPixelDigiSource
*/
  std::cout << "		- " << std::setw(40) << "SiPixelRecHitSource" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiPixelRecHitSource"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiPixelRecHitSource"];
  cumulativetimepermodule += timingPerModule["SiPixelRecHitSource"];
/*
			- SiPixelRecHitSource
*/
  std::cout << "		- " << std::setw(40) << "SiPixelClusterSource" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiPixelClusterSource"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiPixelClusterSource"];
  cumulativetimepermodule += timingPerModule["SiPixelClusterSource"];
/*
			- SiPixelClusterSource
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;

  std::cout << "		2.2.3 : Track monitor " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SiPixelTrackResidualSource" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiPixelTrackResidualSource"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiPixelTrackResidualSource"];
  cumulativetimepermodule += timingPerModule["SiPixelTrackResidualSource"];
/*
			- SiPixelTrackResidualSource
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;

  std::cout << "		2.2.4 : Pixel EDA client " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SiPixelEDAClient" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiPixelEDAClient"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiPixelEDAClient"];
  cumulativetimepermodule += timingPerModule["SiPixelEDAClient"];
/*
			- sipixelEDAClientP5
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;

  std::cout << "Total nb of events read = " << nbofevts << std::endl;
  std::cout << "Total time = " << totalTime << " ms/event " << std::endl;

  std::map<unsigned,double>::const_iterator eventIt = timingPerEvent.begin();
  std::map<unsigned,double>::const_iterator eventEnd = timingPerEvent.end();
  double minEv = 99999999.;
  double maxEv = 0.;
  double rms = 0.;
  double mean = 0.;
  double timeEv = 0;
  for ( ; eventIt != eventEnd; ++eventIt ) { 
    if ( eventIt->first == 1 ) continue;
    timeEv = eventIt->second;
    //std::cout << "Evt nr : " << eventIt->first << " / " << timeEv << " ms" << std::endl;
    if ( timeEv > maxEv ) maxEv = timeEv;
    if ( timeEv < minEv ) minEv = timeEv;
    mean += timeEv;
    rms  += timeEv*timeEv;    
  }

  mean /= ((double)nbofevts-1.);
  rms  /= ((double)nbofevts-1.);
  rms = std::sqrt(rms-mean*mean);
  std::cout << "Total time = " << mean << " +/- " << rms << " ms/event" << std::endl;
  std::cout << "Min.  time = " << minEv << " ms/event" << std::endl;
  std::cout << "Max.  time = " << maxEv << " ms/event" << std::endl;
}

