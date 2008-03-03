
/*
 * $Date: 2008/02/28 14:21:34 $
 *
 * \author: Evelyne Delmeire
 */

#include "AnalysisExamples/SiStripDetectorPerformance/test/SiStripInfoAnalysis/ClusterInfoAnalyzerExample.h"


#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "AnalysisDataFormats/TrackInfo/src/TrackInfo.cc"

#include "sstream"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3S.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TPostScript.h"


  
//---------------------------------------------------------------------------------------------
ClusterInfoAnalyzerExample::ClusterInfoAnalyzerExample(edm::ParameterSet const& conf): 
  conf_(conf),
  theCMNSubtractionMode(conf.getParameter<std::string>( "CMNSubtractionMode" )), 
  theTrackSourceLabel(conf.getParameter<edm::InputTag>( "trackSourceLabel" )), 
  theTrackTrackInfoAssocLabel(conf.getParameter<edm::InputTag>( "trackTrackInfoAssocLabel" )),  
  theClusterSourceLabel(conf.getParameter<edm::InputTag>( "clusterSourceLabel" )),        
  theModulesToBeExcluded(conf.getParameter< std::vector<uint32_t> >("modulesToBeExcluded")),
  daqMonInterface_(edm::Service<DQMStore>().operator->()), 
  show_mechanical_structure_view(true), 
  reset_each_run(false)
{  
  totalNumberOfClustersOnTrack=0;
  countOn=0;
}

//---------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------
ClusterInfoAnalyzerExample::~ClusterInfoAnalyzerExample(){ }  
//---------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------
void ClusterInfoAnalyzerExample::beginJob( const edm::EventSetup& es ) {
  
  es.get<TrackerDigiGeometryRecord>().get( trackerGeom_ ); 
     
  es.get<SiStripDetCablingRcd>().get(stripDetCabling_);
    
  show_mechanical_structure_view = conf_.getParameter<bool>("ShowMechanicalStructureView");
  reset_each_run                 = conf_.getParameter<bool>("ResetMEsEachRun");
  
} // beginJob
//---------------------------------------------------------------------------------------------



//---------------------------------------------------------------------------------------------  
void ClusterInfoAnalyzerExample::endJob(void){
  
  //daqMonInterface_->showDirStructure();
  
  bool putOutputMEsInRootFile   = conf_.getParameter<bool>("putOutputMEsInRootFile");
  std::string outputMEsRootFile = conf_.getParameter<std::string>("outputMEsRootFile");
  
    

  if(putOutputMEsInRootFile){
    
    std::ofstream monitor_summary("monitor_clusterInfo_summary.txt");
    monitor_summary << "The total number of clusters on track = " << totalNumberOfClustersOnTrack<< std::endl;
    monitor_summary << " " << std::endl;   
		    
		    
    monitor_summary <<"ClusterInfoAnalyzerExample::The total number of Monitor Elements = "<< clusterMEs_.size() << std::endl;   
    monitor_summary <<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    
    for(std::map<uint32_t, ModMEs>::const_iterator MEs_iter = clusterMEs_.begin(); 
	                                          MEs_iter!= clusterMEs_.end(); MEs_iter++ ){   
						  
      if((MEs_iter->second).ClusterPosition->getEntries() !=0){	
      					  
        monitor_summary << "ClusterInfoAnalyzerExample" << " " << std::endl;
        monitor_summary << "ClusterInfoAnalyzerExample" << "    ++++++detid  " << MEs_iter->first <<std::endl << std::endl;
      
        monitor_summary << "ClusterInfoAnalyzerExample" << "          +++ ClusterPosition " << std::endl;
	monitor_summary << "ClusterInfoAnalyzerExample" << "          +++      #entries = " << (MEs_iter->second).ClusterPosition->getEntries() << std::endl;
	monitor_summary	<< "ClusterInfoAnalyzerExample" << "          +++          mean = " << (MEs_iter->second).ClusterPosition->getMean()    << std::endl;
	monitor_summary	<< "ClusterInfoAnalyzerExample" << "          +++           RMS = " << (MEs_iter->second).ClusterPosition->getRMS()     << std::endl;
     }    
    } // MEs_iter
    
    monitor_summary << "ClusterInfoAnalyzerExample" << std::endl;            
    
    monitor_summary<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    
    daqMonInterface_->save(outputMEsRootFile);
    
  }//putOutputMEsInRootFile
  
}// endJob
//---------------------------------------------------------------------------------------------  


//--------------------------------------------------------------------------------------------
void ClusterInfoAnalyzerExample::beginRun(const edm::Run& run, const edm::EventSetup& es){
  
  if (show_mechanical_structure_view) {
    
    unsigned long long theCacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
    
    if (cacheID_ != theCacheID) {
      
      cacheID_ = theCacheID;       
      createMEs(es);
    } 
  } 
  else if (reset_each_run) {
        
    for (std::map<uint32_t, ModMEs >::const_iterator MEs_iter = clusterMEs_.begin() ; 
	                                            MEs_iter!= clusterMEs_.end() ; MEs_iter++) {
      ResetModuleMEs(MEs_iter->first);
    }
    
  } //show_mechanical_structure_view
  
} //beginRun

//--------------------------------------------------------------------------------------------
void ClusterInfoAnalyzerExample::endRun(const edm::Run&, const edm::EventSetup&){
}

//--------------------------------------------------------------------------------------------
void ClusterInfoAnalyzerExample::createMEs(const edm::EventSetup& es){
  
  if ( show_mechanical_structure_view ){

    // get list of active detectors from SiStripDetCabling 
    std::vector<uint32_t> activeDets_;
    activeDets_.clear(); // just in case
    stripDetCabling_->addActiveDetectorsRawIds(activeDets_);
    
    
    // safety check : remove any eventual zero elements - there should be none, but just in case
    for(std::vector<uint32_t>::iterator detid_iter = activeDets_.begin(); 
	                               detid_iter != activeDets_.end(); detid_iter++){
      if(*detid_iter == 0) activeDets_.erase(detid_iter);
    }
    
    
    SiStripHistoId hidmanager; //for producing histogram id (and title)
    SiStripFolderOrganizer folder_organizer;    
    folder_organizer.setSiStripFolder();
    
    
    // *** global histograms
								    
    charge_of_each_cluster              = daqMonInterface_->book1D("chargeOfEachCluster",
                                                                   "cluster charge [ADC]",
						                    500,-0.5,500.5); // 31, -0.5, 300.5);
    
    maxCharge_of_each_cluster           = daqMonInterface_->book1D("maxChargeOfEachCluster",
                                                                   "charge of strip with max amplitude [ADC]",
								    500,-0.5,500.5);
								    
    chargeLeft_of_each_cluster          = daqMonInterface_->book1D("chargeLeftOfEachCluster",
                                                                   "charge left to max amplitude strip [ADC]",
								    500,-0.5,500.5);
								    
    chargeRight_of_each_cluster  = daqMonInterface_->book1D("chargeRightOfEachCluster",
                                                                   "charge right to max amplitude strip [ADC]",
								    500,-0.5,500.5);
								    
    width_of_each_cluster               = daqMonInterface_->book1D("widthOfEachCluster",
                                                                   "cluster width [nr strips]",
								    11,-0.5,10.5);
								    
    position_of_each_cluster            = daqMonInterface_->book1D("positionOfEachCluster",
                                                                   "cluster position [strip number +0.5]",
								    24,0,768); // 6 APV's ->768 strips
								    
    maxPosition_of_each_cluster         = daqMonInterface_->book1D("maxPositionOfEachCluster",
                                                                   "position of strip with max amplitude",
								    24,0,768);
								    
    noise_of_each_cluster               = daqMonInterface_->book1D("noiseOfEachCluster",
                                                                   "cluster noise",
								    80,0,10);
								    
    noiseRescaledByGain_of_each_cluster = daqMonInterface_->book1D("noiseRescaledByGainOfEachCluster",
                                                                   "cluster noise rescaled by gain",
								    80,0,10);
								    
    signalOverNoise_of_each_cluster     = daqMonInterface_->book1D("signalOverNoiseOfEachCluster",
                                                                   "cluster signal over noise",
								    100,0,50);
								    
    signalOverNoiseRescaledByGain_of_each_cluster 
                                        = daqMonInterface_->book1D("signalOverNoiseRescaledByGainOfEachCluster",
                                                                   "cluster signal over noise rescaled by gain",
								    100,0,50);
    
 
    
    // *** module-dependent histograms
    //edm::LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::CreateMEs] #activeDets_="<< activeDets_.size();
      
    
    for(std::vector<uint32_t>::const_iterator detid_iter = activeDets_.begin(); 
	                                      detid_iter!=activeDets_.end(); detid_iter++){
      ModMEs clusterME_;
      std::string hid;
      folder_organizer.setDetectorFolder(*detid_iter); // pass the detid to this method
      
      if (reset_each_run) ResetModuleMEs(*detid_iter);
     
      // --> cluster position (i.e.barycenter) 
      hid = hidmanager.createHistoId("ClusterPosition","det",*detid_iter);
      clusterME_.ClusterPosition = daqMonInterface_->book1D(hid, hid, 24,0.,768.); 
      daqMonInterface_->tag(clusterME_.ClusterPosition, *detid_iter); // 6 APVs -> 768 strips
      clusterME_.ClusterPosition->setAxisTitle("cluster barycenter position [strip number +0.5]");
     
       // --> cluster width (i.e.number  of clusterized strips) 
      hid = hidmanager.createHistoId("ClusterWidth","det",*detid_iter);
      clusterME_.ClusterWidth = daqMonInterface_->book1D(hid, hid, 11,-0.5,10.5); 
      daqMonInterface_->tag(clusterME_.ClusterWidth, *detid_iter);
      clusterME_.ClusterWidth->setAxisTitle("cluster width [nr strips]");
          
      // --> cluster charge (also called cluster signal) 
      hid = hidmanager.createHistoId("ClusterCharge","det",*detid_iter);
      clusterME_.ClusterCharge = daqMonInterface_->book1D(hid, hid, 31,-0.5,300.5); 
      daqMonInterface_->tag(clusterME_.ClusterCharge, *detid_iter);
      clusterME_.ClusterCharge->setAxisTitle("cluster charge [ADC]");
      
      // --> position in cluster of strip with maximum amplitude) 
      hid = hidmanager.createHistoId("ClusterMaxPosition","det",*detid_iter);
      clusterME_.ClusterMaxPosition = daqMonInterface_->book1D(hid, hid, 24,0.,768.); 
      daqMonInterface_->tag(clusterME_.ClusterMaxPosition, *detid_iter); // 6 APVs -> 768 strips
      clusterME_.ClusterMaxPosition->setAxisTitle("position in cluster of strip with maximum amplitude [strip number +0.5]");
      
      // --> charge of strip with max amplitude 
      hid = hidmanager.createHistoId("ClusterMaxCharge","det",*detid_iter);
      clusterME_.ClusterMaxCharge = daqMonInterface_->book1D(hid, hid, 31,-0.5,300.5); 
      daqMonInterface_->tag(clusterME_.ClusterMaxCharge, *detid_iter);
      clusterME_.ClusterMaxCharge->setAxisTitle("cluster charge [ADC]");
     
      // --> charge left to max amplitude strip 
      hid = hidmanager.createHistoId("ClusterChargeLeft","det",*detid_iter);
      clusterME_.ClusterChargeLeft = daqMonInterface_->book1D(hid, hid, 31,-0.5,300.5); 
      daqMonInterface_->tag(clusterME_.ClusterChargeLeft, *detid_iter);
      clusterME_.ClusterChargeLeft->setAxisTitle("charge left to max amplitude strip [ADC]");
      
      // --> charge right to max amplitude strip 
      hid = hidmanager.createHistoId("ClusterChargeRight","det",*detid_iter);
      clusterME_.ClusterChargeRight = daqMonInterface_->book1D(hid, hid, 31,-0.5,300.5); 
      daqMonInterface_->tag(clusterME_.ClusterChargeRight, *detid_iter);
      clusterME_.ClusterChargeRight->setAxisTitle("charge right to max amplitude strip[ADC]");
      
      // --> cluster noise 
      hid = hidmanager.createHistoId("ClusterNoise","det",*detid_iter);
      clusterME_.ClusterNoise = daqMonInterface_->book1D(hid, hid, 80,0.,10.); 
      daqMonInterface_->tag(clusterME_.ClusterNoise, *detid_iter);
      clusterME_.ClusterNoise->setAxisTitle("cluster noise");
      
      // --> cluster noise rescaled by gain 
      hid = hidmanager.createHistoId("ClusterNoiseRescaledByGain","det",*detid_iter);
      clusterME_.ClusterNoiseRescaledByGain = daqMonInterface_->book1D(hid, hid, 80,0.,10.); 
      daqMonInterface_->tag(clusterME_.ClusterNoiseRescaledByGain, *detid_iter);
      clusterME_.ClusterNoiseRescaledByGain->setAxisTitle("cluster noise rescaled by gain");

      // --> cluster signal over noise 
      hid = hidmanager.createHistoId("ClusterSignalOverNoise","det",*detid_iter);
      clusterME_.ClusterSignalOverNoise = daqMonInterface_->book1D(hid, hid, 100,0.,50.); 
      daqMonInterface_->tag(clusterME_.ClusterSignalOverNoise, *detid_iter);
      clusterME_.ClusterSignalOverNoise->setAxisTitle("ratio of signal to noise for each cluster");
      
      // --> cluster signal over noise rescaled by gain 
      hid = hidmanager.createHistoId("ClusterSignalOverNoiseRescaledByGain","det",*detid_iter);
      clusterME_.ClusterSignalOverNoiseRescaledByGain = daqMonInterface_->book1D(hid, hid, 100,0.,50.); 
      daqMonInterface_->tag(clusterME_.ClusterSignalOverNoiseRescaledByGain, *detid_iter);
      clusterME_.ClusterSignalOverNoiseRescaledByGain->setAxisTitle("ratio of signal to noise(rescaled by gain) for each cluster");
      
      
      // -->number of clusters per module
      hid = hidmanager.createHistoId("ModuleNrOfClusters","det",*detid_iter);
      clusterME_.ModuleNrOfClusters = daqMonInterface_->book1D(hid, hid, 5,-0.5,4.5); 
      daqMonInterface_->tag(clusterME_.ModuleNrOfClusters, *detid_iter);
      clusterME_.ModuleNrOfClusters->setAxisTitle("number of clusters in one detector module");
      
 
      
      // --> number of clusterized strips per module      
      hid = hidmanager.createHistoId("ModuleNrOfClusterizedStrips","det",*detid_iter);
      clusterME_.ModuleNrOfClusterizedStrips = daqMonInterface_->book1D(hid, hid, 10,-0.5,9.5); 
      daqMonInterface_->tag(clusterME_.ModuleNrOfClusterizedStrips, *detid_iter);
      clusterME_.ModuleNrOfClusterizedStrips->setAxisTitle("number of clusterized strips per module");

       
      // --> ModuleLocalOccupancy per module
      hid = hidmanager.createHistoId("ModuleLocalOccupancy","det",*detid_iter);
      // occupancy goes from 0 to 1, probably not over some limit value (here 0.1)
      clusterME_.ModuleLocalOccupancy = daqMonInterface_->book1D(hid, hid, 20,-0.005,0.05); 
      daqMonInterface_->tag(clusterME_.ModuleLocalOccupancy, *detid_iter);
      clusterME_.ModuleLocalOccupancy->setAxisTitle("module local occupancy [% of clusterized strips]");
      
      
      clusterMEs_.insert( std::make_pair(*detid_iter, clusterME_));
      
    } // loop detid_iter
    
  } // mechanical view
  
} // create ME's
//---------------------------------------------------------------------------------------------  



//---------------------------------------------------------------------------------------------  
void ClusterInfoAnalyzerExample::analyze(const edm::Event& e, const edm::EventSetup& es) { 
     
  runNb   = e.id().run();
  eventNb = e.id().event();
  
  //edm::LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]          Processing RUN=" << runNb << " EVENT=" << eventNb << std::endl;

  
  // -----
  // *** get the collections of track, trackInfos and cluster (clusterInfo=utility class produced on the fly)
  
  // --> SiStripCluster collection
  e.getByLabel( theClusterSourceLabel, clusterHandle_);    
  
  
  // --> trackCollection
  try{ e.getByLabel(theTrackSourceLabel, trackHandle_);
  } catch ( cms::Exception& er ) {
    LogTrace("ClusterInfoAnalyzerExample")<<"caught std::exception Track "<< er.what()<< std::endl;
  } catch ( ... ) {
    LogTrace("ClusterInfoAnalyzerExample")<<" funny error " << std::endl;
  }
  
 
  // --> track-trackInfo association map collection
  
  try{
    e.getByLabel(theTrackTrackInfoAssocLabel,trackTrackInfoAssocHandle_); 
  } catch ( cms::Exception& er ) {
    LogTrace("ClusterInfoAnalyzerExample")<<"caught std::exception TrackInfo "<< er.what()<< std::endl;
  } catch ( ... ) {
    LogTrace("ClusterInfoAnalyzerExample")<<" funny error " << std::endl;
  }
  // -----
  
 
  
  // -----
  // ***  perform basic analysis and fill some cluster-related histograms
    
  const reco::TrackCollection myTracks = *(trackHandle_.product());
  //int nTracks=myTracks.size(); 
  
  
  //edm::LogInfo("ClusterInfoAnalyzerExample")   << "[MYINFO::analyze]          --> #reconstructed tracks="<< nTracks << std::endl ;
  /*
  if(trackHandle_->size()==0){
    edm::LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]          WARNING empty TrackCollection" << std::endl;
  }
  */
  
  // START : loop over tracks
  int iTrack=0;
 
  for (unsigned int myTrack=0; myTrack < trackHandle_->size(); ++myTrack){ // loop over tracks
  
    reco::TrackRef myTrackRef = reco::TrackRef(trackHandle_, myTrack);
    
    /*
    // get basic information about the track from the ref to track   
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]                            "    << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]          *** track number #"    << iTrack+1 << " ***" << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << " " << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]            momentum         : " << myTrackRef->momentum()       << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]            pT               : " << myTrackRef->pt()             << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]            vertex           : " << myTrackRef->vertex()         << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]            impact parameter : " << myTrackRef->d0()             << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]            charge           : " << myTrackRef->charge()         << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]            normalizedChi2   : " << myTrackRef->normalizedChi2() << std::endl;
    edm:: LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]            outer PT         : " << myTrackRef->outerPt()        << std::endl;
    */
    iTrack++;
     
        
    // get additional information about the track from the ref to trackInfo : recHits associated to the track   
    reco::TrackInfoRef myTrackInfoRef = (*trackTrackInfoAssocHandle_.product())[myTrackRef];     
    int recHitsSize = myTrackRef->recHitsSize();   
    //edm::LogInfo("ClusterInfoAnalyzerExample")  << "[MYINFO::analyze]            #recHits="<< recHitsSize << std::endl;
    
           
    // -----
    // minimal track quality criteria : decide whether to keep or to skip the track by check of track quality criteria    
    const edm::ParameterSet theTrackThresholds = conf_.getParameter<edm::ParameterSet>("TrackThresholds");
    
    if( myTrackRef->normalizedChi2() < theTrackThresholds.getParameter<double>("maxChi2") && 
	recHitsSize                  > theTrackThresholds.getParameter<double>("minRecHit") ){
    }
    else{
      /*	
      if( !( myTrackRef->normalizedChi2() < theTrackThresholds.getParameter<double>("maxChi2"))){
       edm::LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]          REASON too large normalized chi2="  << myTrackRef->normalizedChi2() << std::endl;
      }	
      
      if( !(recHitsSize                  > theTrackThresholds.getParameter<double>("minRecHit") )){
       edm::LogInfo("ClusterInfoAnalyzerExample") << "[MYINFO::analyze]          REASON : too few recHits="  << recHitsSize << std::endl;
      }	
      */
      return; 
    }
    // -----
       
    
    // loop over the recHits associated to the track and in order to access corresponding cluster information     
    int iRecHit=0;
    for(reco::TrackInfo::TrajectoryInfo::const_iterator tki_iter = myTrackInfoRef->trajStateMap().begin();
                                                        tki_iter!= myTrackInfoRef->trajStateMap().end();tki_iter++){ // loop over trackInfos
      
      //trajectory local direction and position on detector
      
      LocalVector statedirection;
      LocalPoint  stateposition=(myTrackInfoRef->stateOnDet(Updated,(*tki_iter).first)->parameters()).position();
      
      

      if(myTrackInfoRef->type((*tki_iter).first)==Matched){ 
	
	const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(tki_iter)->first));
	
	if (matchedhit!=0){
	
	            
	  // *** mono side
	  statedirection= myTrackInfoRef->localTrackMomentumOnMono(Updated,(*tki_iter).first);
	  if(statedirection.mag() != 0)         getClusterInfoFromRecHit(matchedhit->monoHit(),statedirection,myTrackRef,es);
	  // *** stereo side
	  statedirection= myTrackInfoRef->localTrackMomentumOnStereo(Updated,(*tki_iter).first);
	  if(statedirection.mag() != 0)         getClusterInfoFromRecHit(matchedhit->stereoHit(),statedirection,myTrackRef,es);
	
	} // non-null hit
	
      } // Matched
      
      else if (myTrackInfoRef->type((*tki_iter).first)==Projected){ 
	
	
	const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(tki_iter)->first));
	
	if(phit!=0){
	

	  // *** mono side
	  statedirection= myTrackInfoRef->localTrackMomentumOnMono(Updated,(*tki_iter).first);
	  if(statedirection.mag() != 0) getClusterInfoFromRecHit(&(phit->originalHit()),statedirection,myTrackRef,es);
	  // *** stereo side
	  statedirection= myTrackInfoRef->localTrackMomentumOnStereo(Updated,(*tki_iter).first);
	  if(statedirection.mag() != 0) getClusterInfoFromRecHit(&(phit->originalHit()),statedirection, myTrackRef,es);
	
	} // non-null hit
	
      } // Projected 
      
      else {
	
	const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(tki_iter)->first));
	
	if(hit!=0){
	  
	       
	  statedirection=(myTrackInfoRef->stateOnDet(Updated,(*tki_iter).first)->parameters()).momentum();
	  if(statedirection.mag() != 0) {
	   getClusterInfoFromRecHit(hit,statedirection,myTrackRef,es);
	  }
	
	} // non-null hit

      } // Dynamic Cast
    iRecHit++;
    } // loop over trackInfos
    
  }
  // STOP : loop over tracks
  // -----



  // analysis of module global occupancy (i.e. not only for on track clusters)
  
      for (std::map<uint32_t, ModMEs>::const_iterator MEs_iter = clusterMEs_.begin() ; 
                                                    MEs_iter!=clusterMEs_.end() ; MEs_iter++) {
      uint32_t detid    = MEs_iter->first;  
      ModMEs clusterME_ = MEs_iter->second;
 
      edm::DetSetVector<SiStripCluster>::const_iterator search_dsv_iter = clusterHandle_->find(detid); // search  clusters of detid
    
      if(search_dsv_iter==clusterHandle_->end()){
        if(clusterME_.ModuleNrOfClusters != NULL){
          (clusterME_.ModuleNrOfClusters)->Fill(0.,1.); // no clusters for this detector module, so fill histogram with 0
        }
        continue; // no clusters for this detid => jump to next step of loop
      }
    
      edm::DetSet<SiStripCluster> clusters_ds_= (*clusterHandle_)[detid]; // the statement above makes sure there exists an element with 'detid'

    
      // *** number of clusters per module
      if(clusterME_.ModuleNrOfClusters != NULL){ 
        (clusterME_.ModuleNrOfClusters)->Fill(static_cast<float>(clusters_ds_.data.size()),1.);
      }
    

      // *** number of clusterized strips per module 
      float total_clusterized_strips = 0;
      for(edm::DetSet<SiStripCluster>::const_iterator clusters_Iter = clusters_ds_.data.begin(); 
                                                      clusters_Iter!= clusters_ds_.data.end(); clusters_Iter++){
        SiStripClusterInfo clusterInfo = SiStripClusterInfo( detid, *clusters_Iter,es, theCMNSubtractionMode);
        total_clusterized_strips = total_clusterized_strips + static_cast<float>(clusterInfo.getWidth()); 
      }     
      if(clusterME_.ModuleNrOfClusterizedStrips != NULL){ 
        (clusterME_.ModuleNrOfClusterizedStrips)->Fill(static_cast<float>(total_clusterized_strips),1.);
      }

      // *** ModuleLocalOccupancy 
      short total_nr_strips = 6 * 128; // assume 6 APVs per detector for the moment
      float local_occupancy = static_cast<float>(total_clusterized_strips)/static_cast<float>(total_nr_strips);
      if(clusterME_.ModuleLocalOccupancy != NULL){ 
        (clusterME_.ModuleLocalOccupancy)->Fill(local_occupancy,1.);
      }
      
  } // clusterMEs_


  
} // analyze 
//---------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------
bool ClusterInfoAnalyzerExample::fillClusterHistos(SiStripClusterInfo*      clusterInfo_,
						   const uint32_t&                detId_,  
						   TString                       flag_,
						   const LocalVector clusterLocalVector_) {
  if (clusterInfo_==0) return false;
  
  const  edm::ParameterSet theClusterConditions = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
    
  if  ( theClusterConditions.getParameter<bool>("On") 
	&&
	( 
	 double(clusterInfo_->getCharge())/ clusterInfo_->getNoise() < theClusterConditions.getParameter<double>("minStoN") 
	 ||
	 double(clusterInfo_->getCharge())/clusterInfo_->getNoise() > theClusterConditions.getParameter<double>("maxStoN") 
	 ||
	 double(clusterInfo_->getWidth()) < theClusterConditions.getParameter<double>("minWidth") 
	 ||
	 double(clusterInfo_->getWidth()) > theClusterConditions.getParameter<double>("maxWidth") 
	 )
	)
  return false;
    
  totalNumberOfClustersOnTrack++;
  
  // global histograms
  
  position_of_each_cluster                      ->Fill( static_cast<float>( clusterInfo_->getPosition())                      ,1.);
  width_of_each_cluster                         ->Fill( static_cast<float>( clusterInfo_->getWidth())                         ,1.);
  charge_of_each_cluster                        ->Fill( static_cast<float>( clusterInfo_->getCharge() )                       ,1.);
  maxPosition_of_each_cluster                   ->Fill( static_cast<float>( clusterInfo_->getMaxPosition())                   ,1.);
  maxCharge_of_each_cluster                     ->Fill( static_cast<float>( clusterInfo_->getMaxCharge() )                    ,1.);
  chargeLeft_of_each_cluster                    ->Fill( static_cast<float>((clusterInfo_->getChargeLR()).first )       ,1.);
  chargeRight_of_each_cluster                   ->Fill( static_cast<float>((clusterInfo_->getChargeLR()).second )      ,1.);
  noise_of_each_cluster                         ->Fill( static_cast<float>( clusterInfo_->getNoise() )                        ,1.);
  noiseRescaledByGain_of_each_cluster           ->Fill( static_cast<float>( clusterInfo_->getNoiseRescaledByGain())           ,1.);
  signalOverNoise_of_each_cluster               ->Fill( static_cast<float>( clusterInfo_->getSignalOverNoise() )              ,1.);
  signalOverNoiseRescaledByGain_of_each_cluster ->Fill( static_cast<float>( clusterInfo_->getSignalOverNoiseRescaledByGain()) ,1.);
 
  // module-dependent histograms
  // fill the monitoring element corresponding to the detid_ of cluster
    
  for (std::map<uint32_t, ModMEs>::const_iterator MEs_iter = clusterMEs_.begin(); 
                                                 MEs_iter!= clusterMEs_.end() ; MEs_iter++) {

    if( (MEs_iter->first)!= detId_ ) continue; // fill ME corresponding to out detId
    ModMEs clusterME_ = MEs_iter->second;


    // safety check: get from DetSetVector the DetSet of clusters belonging to one detId_ 
    //               sand make sure there exists clusters with this id
    
    edm::DetSetVector<SiStripCluster>::const_iterator search_dsv_iter = clusterHandle_->find(detId_);  
    if(search_dsv_iter==clusterHandle_->end()){
      continue;
    }
 

    edm::DetSet<SiStripCluster> clusters_ds_ = (*clusterHandle_)[detId_]; // the statement above makes  
                                                                             // sure there exists 
                                                                             // an element with 'detId_'
     // *** cluster width (i.e.number of clusterized strips)     
    if(clusterME_.ClusterWidth != NULL){ 
      (clusterME_.ClusterWidth)->Fill( static_cast<float>( clusterInfo_->getWidth())               ,1.);
    }

    // *** cluster position (barycenter)     
    if(clusterME_.ClusterPosition != NULL){ 
      (clusterME_.ClusterPosition)->Fill( static_cast<float>( clusterInfo_->getPosition())            ,1.);
    }

    // *** cluster charge also called cluster signal
    if(clusterME_.ClusterCharge != NULL){ 
      (clusterME_.ClusterCharge)->Fill( static_cast<float>( clusterInfo_->getCharge() )             ,1.);
    }

    // *** cluster maximum position (i.e. position strip with maximum amplitude)     
    if(clusterME_.ClusterMaxPosition != NULL){ 
      (clusterME_.ClusterMaxPosition)->Fill( static_cast<float>( clusterInfo_->getMaxPosition())         ,1.);
    }

     // *** charge of strip with max amplitude     
    if(clusterME_.ClusterMaxCharge != NULL){ 
      (clusterME_.ClusterMaxCharge)->Fill( static_cast<float>( clusterInfo_->getMaxCharge() )           ,1.);
    }

    // *** charge left to max amplitude strip     
    if(clusterME_.ClusterChargeLeft != NULL){ 
     (clusterME_.ClusterChargeLeft)->Fill( static_cast<float>((clusterInfo_->getChargeLR()).first )    ,1.);
    }

    // *** charge right to max amplitude strip     
    if(clusterME_.ClusterChargeRight != NULL){ 
      (clusterME_.ClusterChargeRight)->Fill( static_cast<float>((clusterInfo_->getChargeLR()).second )   ,1.);
    }
   
    // *** cluster noise     
    if(clusterME_.ClusterNoise != NULL){ 
      (clusterME_.ClusterNoise)->Fill( static_cast<float>( clusterInfo_->getNoise() )              ,1.);
    }

    // *** cluster noise rescaled by gain     
    if(clusterME_.ClusterNoiseRescaledByGain != NULL){ 
      (clusterME_.ClusterNoiseRescaledByGain)->Fill( static_cast<float>( clusterInfo_->getNoiseRescaledByGain()) ,1.);
    }

    // *** cluster signal over noise     
    if(clusterME_.ClusterSignalOverNoise != NULL){ 
      (clusterME_.ClusterSignalOverNoise)->Fill( static_cast<float>(clusterInfo_->getSignalOverNoise()) ,1.);
    }

    // *** cluster signal over noise rescaled by gain 
    if(clusterME_.ClusterSignalOverNoiseRescaledByGain != NULL){ 
      (clusterME_.ClusterSignalOverNoiseRescaledByGain)->Fill( static_cast<float>(clusterInfo_->getSignalOverNoiseRescaledByGain()) ,1.);
    }      

  } // clusterMEs_

   return true;
  
} // fillClusterHistos
//---------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------
void ClusterInfoAnalyzerExample::getClusterInfoFromRecHit(const SiStripRecHit2D* trackerRecHit_, 
						          LocalVector      clusterLocalVector_,
						          reco::TrackRef             trackRef_,
						          const edm::EventSetup&            es){
  
  // method that allows to get the cluster associated to the recHit and 
  // that adds/stores the pointer to it in a vector of SiStripCluster pointers for each track in the event
 
      
  // *** check hit validity
  
  if(!trackerRecHit_->isValid()){
    return;  
  }
 
  const uint32_t& recHitDetId = trackerRecHit_->geographicalId().rawId();
  
  if (find(theModulesToBeExcluded.begin(),theModulesToBeExcluded.end(),recHitDetId)!=theModulesToBeExcluded.end()){
    return;
  }

  /*
  edm::LogInfo("ClusterInfoAnalyzerExample") 
    << "[MYINFO::fillClusterInfo]  RecHit on det             : " << trackerRecHit_->geographicalId().rawId() << std::endl;
  edm::LogInfo("ClusterInfoAnalyzerExample") 
    << "[MYINFO::fillClusterInfo]  RecHit in LP              : " << trackerRecHit_->localPosition() << std::endl;
  edm::LogInfo("ClusterInfoAnalyzerExample") 
    << "[MYINFO::fillClusterInfo]  RecHit in GP              : " 
    << trackerGeom_->idToDet(trackerRecHit_->geographicalId())->surface().toGlobal(trackerRecHit_->localPosition())   << std::endl;
   edm::LogInfo("ClusterInfoAnalyzerExample") 
    << "[MYINFO::fillClusterInfo]  RecHit track local vector : " << clusterLocalVector_.x() 
					                                    << " " << clusterLocalVector_.y() 
									    << " " << clusterLocalVector_.z() << std::endl; 
  */
  
  // now get the cluster from RecHit and produce clusterInfo on the fly
  
  if ( trackerRecHit_ != NULL ){
    
    SiStripClusterInfo clusterInfo = SiStripClusterInfo( recHitDetId, *(trackerRecHit_->cluster()),es, theCMNSubtractionMode);
    
 
    // additional cross check for safety although track should have been excluded before getting here
    
    const edm::ParameterSet theTrackThresholds = conf_.getParameter<edm::ParameterSet>("TrackThresholds");
    if( trackRef_->normalizedChi2() < theTrackThresholds.getParameter<double>("maxChi2") && 
	trackRef_->recHitsSize()    > theTrackThresholds.getParameter<double>("minRecHit") ){
   
      if ( fillClusterHistos( & clusterInfo, recHitDetId ,"_onTrack", clusterLocalVector_ ) ) {
         
 	countOn++;
	
      } // fill histos for clusters satisfying cluster conditions
      
    } //additional cross check  
  } // non-NULL recHit
  
} // fillClusterInfo
//---------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------
void ClusterInfoAnalyzerExample::ResetME(MonitorElement* me){
  if (me && me->kind() == MonitorElement::DQM_KIND_TH1F) {
    TH1F * root_ob = me->getTH1F();
    if(root_ob) root_ob->Reset();
  }
}
//-----------------

//---------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------

void ClusterInfoAnalyzerExample::ResetModuleMEs(uint32_t idet){
  std::map<uint32_t, ModMEs >::iterator MEs_iter = clusterMEs_.find(idet);
  ModMEs clusterME_ = MEs_iter->second;
  
  ResetME( clusterME_.ClusterWidth);
  ResetME( clusterME_.ClusterPosition);
  ResetME( clusterME_.ClusterMaxPosition);
  ResetME( clusterME_.ClusterCharge);
  ResetME( clusterME_.ClusterMaxCharge);
  ResetME( clusterME_.ClusterChargeLeft);
  ResetME( clusterME_.ClusterChargeRight);
  ResetME( clusterME_.ClusterNoise);
  ResetME( clusterME_.ClusterNoiseRescaledByGain);
  ResetME( clusterME_.ClusterSignalOverNoise);
  ResetME( clusterME_.ClusterSignalOverNoiseRescaledByGain);
  ResetME( clusterME_.ModuleNrOfClusters);
  ResetME( clusterME_.ModuleNrOfClusterizedStrips);
  ResetME( clusterME_.ModuleLocalOccupancy);

  
  
}

//---------------------------------------------------------------------------------------------



  

