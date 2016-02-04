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


  std::cout << "	1.2 BeamSpot+CkfTracks :" << std::endl;
  std::cout << "		1.2.1 : BeamSpot " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "BeamSpotProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["BeamSpotProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["BeamSpotProducer"];
  cumulativetimepermodule += timingPerModule["BeamSpotProducer"];
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		1.2.2 : Tracks " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "CosmicSeedGenerator" << std::setw(30) << "" << std::setw(8) << timingPerModule["CosmicSeedGenerator"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["CosmicSeedGenerator"];
  cumulativetimepermodule += timingPerModule["CosmicSeedGenerator"];
/*
			- cosmicseedfinderP5
			- cosmicseedfinderP5Top
			- cosmicseedfinderP5Bottom
*/
  std::cout << "		- " << std::setw(40) << "SimpleCosmicBONSeeder" << std::setw(30) << "" << std::setw(8) << timingPerModule["SimpleCosmicBONSeeder"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SimpleCosmicBONSeeder"];
  cumulativetimepermodule += timingPerModule["SimpleCosmicBONSeeder"];
/*
			- simpleCosmicBONSeeds
			- simpleCosmicBONSeedsTop
			- simpleCosmicBONSeedsBottom
*/
  std::cout << "		- " << std::setw(40) << "SeedCombiner" << std::setw(30) << "" << std::setw(8) << timingPerModule["SeedCombiner"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SeedCombiner"];
  cumulativetimepermodule += timingPerModule["SeedCombiner"];
/*
			- combinedP5SeedsForCTF
			- combinedP5SeedsForCTFTop
			- combinedP5SeedsForCTFBottom
*/
  std::cout << "		- " << std::setw(40) << "CtfSpecialSeedGenerator" << std::setw(30) << "" << std::setw(8) << timingPerModule["CtfSpecialSeedGenerator"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["CtfSpecialSeedGenerator"];
  cumulativetimepermodule += timingPerModule["CtfSpecialSeedGenerator"];
/*
			- combinatorialcosmicseedfinderP5
			- combinatorialcosmicseedfinderP5Top
			- combinatorialcosmicseedfinderP5Bottom
*/
  std::cout << "		- " << std::setw(40) << "RoadSearchSeedFinder" << std::setw(30) << "" << std::setw(8) << timingPerModule["RoadSearchSeedFinder"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["RoadSearchSeedFinder"];
  cumulativetimepermodule += timingPerModule["RoadSearchSeedFinder"];
/*
			- roadSearchSeedsP5
			- roadSearchSeedsP5Top
			- roadSearchSeedsP5Bottom
*/
  std::cout << "		- " << std::setw(40) << "RoadSearchCloudMaker" << std::setw(30) << "" << std::setw(8) << timingPerModule["RoadSearchCloudMaker"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["RoadSearchCloudMaker"];
  cumulativetimepermodule += timingPerModule["RoadSearchCloudMaker"];
/*
			- roadSearchCloudsP5
			- roadSearchCloudsP5Top
			- roadSearchCloudsP5Bottom
*/
  std::cout << "		- " << std::setw(40) << "CosmicTrackFinder" << std::setw(30) << "" << std::setw(8) << timingPerModule["CosmicTrackFinder"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["CosmicTrackFinder"];
  cumulativetimepermodule += timingPerModule["CosmicTrackFinder"];
/*
			- cosmicCandidateFinderP5
			- cosmicCandidateFinderP5Top
			- cosmicCandidateFinderP5Bottom
*/
  std::cout << "		- " << std::setw(40) << "CosmicTrackSplitter" << std::setw(30) << "" << std::setw(8) << timingPerModule["CosmicTrackSplitter"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["CosmicTrackSplitter"];
  cumulativetimepermodule += timingPerModule["CosmicTrackSplitter"];
/*
			- cosmicTrackSplitter
*/
  std::cout << "		- " << std::setw(40) << "CkfTrackCandidateMaker" << std::setw(30) << "" << std::setw(8) << timingPerModule["CkfTrackCandidateMaker"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["CkfTrackCandidateMaker"];
  cumulativetimepermodule += timingPerModule["CkfTrackCandidateMaker"];
/*
			- ckfTrackCandidatesP5
			- ckfTrackCandidatesP5LHCNavigation
			- ckfTrackCandidatesP5Top
			- ckfTrackCandidatesP5Bottom
*/
  std::cout << "		- " << std::setw(40) << "RoadSearchTrackCandidateMaker" << std::setw(30) << "" << std::setw(8) << timingPerModule["RoadSearchTrackCandidateMaker"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["RoadSearchTrackCandidateMaker"];
  cumulativetimepermodule += timingPerModule["RoadSearchTrackCandidateMaker"];
/*
			- rsTrackCandidatesP5
			- rsTrackCandidatesP5Top
			- rsTrackCandidatesP5Bottom
*/
  std::cout << "		- " << std::setw(40) << "TrackProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["TrackProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["TrackProducer"];
  cumulativetimepermodule += timingPerModule["TrackProducer"];
/*
			- cosmictrackfinderP5
			- cosmictrackfinderP5Top
			- cosmictrackfinderP5Bottom
			- splittedTracksP5
			- ctfWithMaterialTracksP5LHCNavigation
			- ctfWithMaterialTracksP5
			- ctfWithMaterialTracksP5Top
			- ctfWithMaterialTracksP5Bottom
			- rsWithMaterialTracksP5
			- rsWithMaterialTracksP5Top
			- rsWithMaterialTracksP5Bottom
*/
  std::cout << "		- " << std::setw(40) << "PixelClusterSelectorTopBottom" << std::setw(30) << "" << std::setw(8) << timingPerModule["PixelClusterSelectorTopBottom"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["PixelClusterSelectorTopBottom"];
  cumulativetimepermodule += timingPerModule["PixelClusterSelectorTopBottom"];
/*
			- siPixelClustersTop
			- siPixelClustersBottom
*/
  std::cout << "		- " << std::setw(40) << "SiPixelRecHitConverter" << std::setw(30) << "" << std::setw(8) << timingPerLabel["SiPixelRecHitConverter:siPixelRecHitsTop"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiPixelRecHitConverter:siPixelRecHitsBottom"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerLabel["SiPixelRecHitConverter:siPixelRecHitsTop"];
  subtotaltimepermodule   += timingPerLabel["SiPixelRecHitConverter:siPixelRecHitsBottom"];
  cumulativetimepermodule += timingPerLabel["SiPixelRecHitConverter:siPixelRecHitsTop"];
  cumulativetimepermodule += timingPerLabel["SiPixelRecHitConverter:siPixelRecHitsBottom"];
/*
			- siPixelRecHitsTop
			- siPixelRecHitsBottom
*/
  std::cout << "		- " << std::setw(40) << "StripClusterSelectorTopBottom" << std::setw(30) << "" << std::setw(8) << timingPerModule["StripClusterSelectorTopBottom"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["StripClusterSelectorTopBottom"];
  cumulativetimepermodule += timingPerModule["StripClusterSelectorTopBottom"];
/*
			- siStripClustersTop
			- siStripClustersBottom
*/
  std::cout << "		- " << std::setw(40) << "SiStripRecHitConverter" << std::setw(30) << "" << std::setw(8) << timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHitsTop"]/((double)nbofevts-1.)
  															  +timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHitsBottom"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHitsTop"];
  subtotaltimepermodule   += timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHitsBottom"];
  cumulativetimepermodule += timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHitsTop"];
  cumulativetimepermodule += timingPerLabel["SiStripRecHitConverter:siStripMatchedRecHitsBottom"];
/*
			- siStripMatchedRecHitsTop
			- siStripMatchedRecHitsBottom
*/
  std::cout << "		- " << std::setw(40) << "TopBottomClusterInfoProducer" << std::setw(30) << "" << std::setw(8) << timingPerModule["TopBottomClusterInfoProducer"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["TopBottomClusterInfoProducer"];
  cumulativetimepermodule += timingPerModule["TopBottomClusterInfoProducer"];
/*
			- topBottomClusterInfoProducerBottom
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
  std::cout << "		- " << std::setw(40) << "" << std::setw(30) << "" << std::setw(8) << timingPerModule["QualityTester"]/((double)nbofevts-1.) << " ms/event" << std::endl;
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
  std::cout << "		2.1.3 : DQM playback for Tracking info " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "DQMEventInfo" << std::setw(30) << "" << std::setw(8) << timingPerLabel["DQMEventInfo:dqmEnvTr"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerLabel["DQMEventInfo:dqmEnvTr"];
  cumulativetimepermodule += timingPerLabel["DQMEventInfo:dqmEnvTr"];
/*
			- dqmEnvTr
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		2.1.4 : DQM file saver " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "DQMFileSaver" << std::setw(30) << "" << std::setw(8) << timingPerModule["DQMFileSaver"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["DQMFileSaver"];
  cumulativetimepermodule += timingPerModule["DQMFileSaver"];
/*
			- dqmSaver
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;

  std::cout << "	2.2 DQM monitor " << std::endl;
  std::cout << "		2.2.1 : SiStripMonitor " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SiStripFEDMonitorPlugin" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripFEDMonitorPlugin"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiStripFEDMonitorPlugin"];
  cumulativetimepermodule += timingPerModule["SiStripFEDMonitorPlugin"];
/*
			- siStripFEDMonitor
*/
  std::cout << "		- " << std::setw(40) << "SiStripMonitorDigi" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripMonitorDigi"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiStripMonitorDigi"];
  cumulativetimepermodule += timingPerModule["SiStripMonitorDigi"];
/*
			- SiStripMonitorDigi
*/
  std::cout << "		- " << std::setw(40) << "SiStripMonitorCluster" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripMonitorCluster"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiStripMonitorCluster"];
  cumulativetimepermodule += timingPerModule["SiStripMonitorCluster"];
/*
			- SiStripMonitorClusterReal
*/
  std::cout << "		- " << std::setw(40) << "SiStripMonitorTrack" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripMonitorTrack"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiStripMonitorTrack"];
  cumulativetimepermodule += timingPerModule["SiStripMonitorTrack"];
/*
			- SiStripMonitorTrack_cosmicTk
*/
  std::cout << "		- " << std::setw(40) << "MonitorTrackResiduals" << std::setw(30) << "" << std::setw(8) << timingPerModule["MonitorTrackResiduals"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["MonitorTrackResiduals"];
  cumulativetimepermodule += timingPerModule["MonitorTrackResiduals"];
/*
			- MonitorTrackResiduals_cosmicTk
*/
  std::cout << "		- " << std::setw(40) << "TrackingMonitor" << std::setw(30) << "" << std::setw(8) << timingPerModule["TrackingMonitor"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["TrackingMonitor"];
  cumulativetimepermodule += timingPerModule["TrackingMonitor"];
/*
			- TrackMon_cosmicTk
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;
  std::cout << "		2.2.2 : SiStripAnalyser " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "SiStripAnalyser" << std::setw(30) << "" << std::setw(8) << timingPerModule["SiStripAnalyser"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["SiStripAnalyser"];
  cumulativetimepermodule += timingPerModule["SiStripAnalyser"];
/*
			- SiStripAnalyser
*/
  std::cout << "		  " << std::setw(70) << "" << std::setw(8) << "--------" << std::endl;
  std::cout << "		  " << std::setw(70) << "subtotal : " << std::setw(8) << subtotaltimepermodule/((double)nbofevts-1.) << " ms/event" << " / " << std::setw(8) << cumulativetimepermodule/((double)nbofevts-1.) << " ms/event" << std::endl;

  std::cout << "		2.2.3 : Miscellaneous " << std::endl;
  subtotaltimepermodule  = 0;
  std::cout << "		- " << std::setw(40) << "TriggerResultInserter" << std::setw(30) << "" << std::setw(8) << timingPerModule["TriggerResultInserter"]/((double)nbofevts-1.) << " ms/event" << std::endl;
  subtotaltimepermodule   += timingPerModule["TriggerResultInserter"];
  cumulativetimepermodule += timingPerModule["TriggerResultInserter"];
/*
			- TriggerResults
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

