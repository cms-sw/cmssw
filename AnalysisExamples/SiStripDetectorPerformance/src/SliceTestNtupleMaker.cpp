#include <functional>

#include <TFile.h>
#include <TTree.h>

#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/GetSiStripClusterXTalk.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/MapHITTSOS.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackUpdatedTSOS.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Handle.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/SliceTestNtupleMaker.h"

SliceTestNtupleMaker::SliceTestNtupleMaker( const edm::ParameterSet &roCONFIG):
  // Read configuration
  oOFileName_( 
    roCONFIG.getUntrackedParameter<std::string>( 
      "oOFileName", 
      "SliceTestNtupleMaker_out.root")) {

  // Extract Digis Label
  try {
    LogDebug( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "\t* Extract Digis Label";

    oConfig_.oITDigi = 
      roCONFIG.getUntrackedParameter<edm::InputTag>( "oDigi");
  } catch( edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "Failed to extract 'oDigi' config value. Check if it is specified "
      << "in config file.";

    // Pass Exception on
    throw;
  }

  // Extract Clusters Label
  try {
    LogDebug( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "\t* Extract Clusters Label";

    oConfig_.oITCluster = 
      roCONFIG.getUntrackedParameter<edm::InputTag>( "oCluster");
  } catch( edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "Failed to extract 'oCluster' config value. Check if it is specified "
      << "in config file.";

    // Pass Exception on
    throw;
  }

  // Extract ClusterInfos Label
  try {
    LogDebug( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "\t* Extract ClusterInfos Label";

    oConfig_.oITClusterInfo = 
      roCONFIG.getUntrackedParameter<edm::InputTag>( "oClusterInfo");
  } catch( edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "Failed to extract 'oClusterInfo' config value. Check if it is "
      << "specified in config file.";

    // Pass Exception on
    throw;
  }

  // Extract Tracks Label
  try {
    LogDebug( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "\t* Extract Tracks Label";

    oConfig_.oITTrack = 
      roCONFIG.getUntrackedParameter<edm::InputTag>( "oTrack");
  } catch( edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "Failed to extract 'oTrack' config value. Check if it is specified "
      << "in config file.";

    // Pass Exception on
    throw;
  }

  // Extract Transient Builder Label
  try {
    LogDebug( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "\t* Extract Transient Builder Label";

    oConfig_.oTTRHBuilder = 
      roCONFIG.getUntrackedParameter<std::string>( "oTTRHBuilder");
  } catch( edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::SliceTestNtupleMaker")
      << "Failed to extract 'oTTRHBuilder' config value. Check if it is "
      << "specified in config file.";

    // Pass Exception on
    throw;
  }
}

// +---------------------------------------------------------------------------
// |  PROTECTED
// +---------------------------------------------------------------------------
void SliceTestNtupleMaker::beginJob( const edm::EventSetup &roEVENT_SETUP) {
  // Reopen output file
  // [Note: object will be destroyed automatically due to std::auto_ptr<...>]
  poOFile_ = std::auto_ptr<TFile>( new TFile( oOFileName_.c_str(), "RECREATE"));

  // Create Cluster Tree and reserve leafs in it
  // [Note: object will be destroyed automatically once Output file is closed]
  poClusterTree_ = new TTree( "ClusterTree", "Cluster Tree");
  poClusterTree_->Branch( "nRun",         &oGenVal_.nRun,        "nRun/I");
  poClusterTree_->Branch( "nLclEvent",    &oGenVal_.nLclEvent,   "nLclEvent/I");
  poClusterTree_->Branch( "nLTime",       &oGenVal_.nLTime,      "nLTime/I");

                                        // Track
  poClusterTree_->Branch( "bTrack",       &oTrackVal_.bTrack,    "bTrack/O");

                                        // Hits
  poClusterTree_->Branch( "dHitLclX",     
                          &oHitVal_.oCoordsLocal.dX, "dHitLclX/F");
  poClusterTree_->Branch( "dHitLclY",     
                          &oHitVal_.oCoordsLocal.dY, "dHitLclY/F");
  poClusterTree_->Branch( "dHitLclZ",     
                          &oHitVal_.oCoordsLocal.dZ, "dHitLclZ/F");
  poClusterTree_->Branch( "dHitLclR",     
                          &oHitVal_.oCoordsLocal.dR, "dHitLclR/F");
  poClusterTree_->Branch( "dHitLclPhi",     
                          &oHitVal_.oCoordsLocal.dPhi, "dHitLclPhi/F");
  poClusterTree_->Branch( "dHitLclMag",     
                          &oHitVal_.oCoordsLocal.dMag, "dHitLclMag/F");
  poClusterTree_->Branch( "dHitLclTheta",     
                          &oHitVal_.oCoordsLocal.dTheta, "dHitLclTheta/F");
  poClusterTree_->Branch( "bHitMatched",
                          &oHitVal_.bMatched, "bHitMatched/O");

  poClusterTree_->Branch( "dHitGlbX",     
                          &oHitVal_.oCoordsGlobal.dX, "dHitGlbX/F");
  poClusterTree_->Branch( "dHitGlbY",     
                          &oHitVal_.oCoordsGlobal.dY, "dHitGlbY/F");
  poClusterTree_->Branch( "dHitGlbZ",     
                          &oHitVal_.oCoordsGlobal.dZ, "dHitGlbZ/F");
  poClusterTree_->Branch( "dHitGlbR",     
                          &oHitVal_.oCoordsGlobal.dR, "dHitGlbR/F");
  poClusterTree_->Branch( "dHitGlbPhi",     
                          &oHitVal_.oCoordsGlobal.dPhi, "dHitGlbPhi/F");
  poClusterTree_->Branch( "dHitGlbMag",     
                          &oHitVal_.oCoordsGlobal.dMag, "dHitGlbMag/F");
  poClusterTree_->Branch( "dHitGlbTheta",     
                          &oHitVal_.oCoordsGlobal.dTheta, "dHitGlbTheta/F");

  poClusterTree_->Branch( "nHitDetSubDet",       
                          &oHitVal_.oDetVal.eSubDet,     
                          "nHitDetSubDet/I");
  poClusterTree_->Branch( "nHitDetLayer",        
                          &oHitVal_.oDetVal.nLayer,      
                          "nHitDetLayer/I");
  poClusterTree_->Branch( "nHitDetWheel",        
                          &oHitVal_.oDetVal.nWheel,      
                          "nHitDetWheel/I");
  poClusterTree_->Branch( "nHitDetRing",         
                          &oHitVal_.oDetVal.nRing,       
                          "nHitDetRing/I");
  poClusterTree_->Branch( "nHitDetSide",         
                          &oHitVal_.oDetVal.nSide, 
                          "nHitDetSide/I");
  poClusterTree_->Branch( "nHitDetPetal",        
                          &oHitVal_.oDetVal.nPetal, 
                          "nHitDetPetal/I");
  poClusterTree_->Branch( "nHitDetPetalFwBw",    
                          &oHitVal_.oDetVal.nPetalFwBw, 
                          "nHitDetPetalFwBw/I");
  poClusterTree_->Branch( "nHitDetRod",          
                          &oHitVal_.oDetVal.nRod, 
                          "nHitDetRod/I");
  poClusterTree_->Branch( "nHitDetRodFwBw",      
                          &oHitVal_.oDetVal.nRodFwBw, 
                          "nHitDetRodFwBw/I");
  poClusterTree_->Branch( "nHitDetString",       
                          &oHitVal_.oDetVal.nString, 
                          "nHitDetString/I");
  poClusterTree_->Branch( "nHitDetStringFwBw",   
                          &oHitVal_.oDetVal.nStringFwBw, 
                          "nHitDetStringFwBw/I");
  poClusterTree_->Branch( "nHitDetStringIntExt", 
                          &oHitVal_.oDetVal.nStringIntExt, 
                          "nHitDetStringIntExt/I");
  poClusterTree_->Branch( "bHitDetStereo",       
                          &oHitVal_.oDetVal.bStereo, 
                          "bHitDetStereo/O");
  poClusterTree_->Branch( "nHitDetModule",       
                          &oHitVal_.oDetVal.nModule,     
                          "nHitDetModule/I");
  poClusterTree_->Branch( "nHitDetModuleFwBw",   
                          &oHitVal_.oDetVal.nModuleFwBw, 
                          "nHitDetModuleFwBw/I");

                                        // Clusters
  poClusterTree_->Branch( "nClusterFirstStrip", 
                          &oClusterVal_.nFirstStrip, "nClusterFirstStrip/I");
  poClusterTree_->Branch( "dClusterCharge", 
                          &oClusterVal_.dCharge, "dClusterCharge/F");
  poClusterTree_->Branch( "dClusterChargeL", 
                          &oClusterVal_.dChargeL, "dClusterChargeL/F");
  poClusterTree_->Branch( "dClusterChargeR", 
                          &oClusterVal_.dChargeR, "dClusterChargeR/F");
  poClusterTree_->Branch( "dClusterNoise",
                          &oClusterVal_.dNoise, "dClusterNoise/F");
  poClusterTree_->Branch( "dClusterBaryCenter",
                          &oClusterVal_.dBaryCenter, "dClusterBaryCenter/F");
  poClusterTree_->Branch( "dClusterXTalk", 
                          &oClusterVal_.dXTalk, "dClsuterXTalk/F");

  poClusterTree_->Branch( "nClusterDetSubDet",       
                          &oClusterVal_.oDetVal.eSubDet,     
                          "nClusterDetSubDet/I");
  poClusterTree_->Branch( "nClusterDetLayer",        
                          &oClusterVal_.oDetVal.nLayer,      
                          "nClusterDetLayer/I");
  poClusterTree_->Branch( "nClusterDetWheel",        
                          &oClusterVal_.oDetVal.nWheel,      
                          "nClusterDetWheel/I");
  poClusterTree_->Branch( "nClusterDetRing",         
                          &oClusterVal_.oDetVal.nRing,       
                          "nClusterDetRing/I");
  poClusterTree_->Branch( "nClusterDetSide",         
                          &oClusterVal_.oDetVal.nSide, 
                          "nClusterDetSide/I");
  poClusterTree_->Branch( "nClusterDetPetal",        
                          &oClusterVal_.oDetVal.nPetal, 
                          "nClusterDetPetal/I");
  poClusterTree_->Branch( "nClusterDetPetalFwBw",    
                          &oClusterVal_.oDetVal.nPetalFwBw, 
                          "nClusterDetPetalFwBw/I");
  poClusterTree_->Branch( "nClusterDetRod",          
                          &oClusterVal_.oDetVal.nRod, 
                          "nClusterDetRod/I");
  poClusterTree_->Branch( "nClusterDetRodFwBw",      
                          &oClusterVal_.oDetVal.nRodFwBw, 
                          "nClusterDetRodFwBw/I");
  poClusterTree_->Branch( "nClusterDetString",       
                          &oClusterVal_.oDetVal.nString, 
                          "nClusterDetString/I");
  poClusterTree_->Branch( "nClusterDetStringFwBw",   
                          &oClusterVal_.oDetVal.nStringFwBw, 
                          "nClusterDetStringFwBw/I");
  poClusterTree_->Branch( "nClusterDetStringIntExt", 
                          &oClusterVal_.oDetVal.nStringIntExt, 
                          "nClusterDetStringIntExt/I");
  poClusterTree_->Branch( "bClusterDetStereo",       
                          &oClusterVal_.oDetVal.bStereo, 
                          "bClusterDetStereo/O");
  poClusterTree_->Branch( "nClusterDetModule",       
                          &oClusterVal_.oDetVal.nModule,     
                          "nClusterDetModule/I");
  poClusterTree_->Branch( "nClusterDetModuleFwBw",   
                          &oClusterVal_.oDetVal.nModuleFwBw, 
                          "nClusterDetModuleFwBw/I");

  // Create Track Tree and reserve leafs in it
  // [Note: object will be destroyed automatically once Output file is closed]
  poTrackTree_ = new TTree( "TrackTree", "Track Tree");
  poTrackTree_->Branch( "nRun",      &oGenVal_.nRun,      "nRun/I");
  poTrackTree_->Branch( "nLclEvent", &oGenVal_.nLclEvent, "nLclEvent/I");
  poTrackTree_->Branch( "nLTime",    &oGenVal_.nLTime,    "nLTime/I");

  poTrackTree_->Branch( "nTrackNum",      &oTrackVal_.nTrackNum, 
                        "nTrackNum/I");
  poTrackTree_->Branch( "dTrackMomentum", &oTrackVal_.dMomentum, 
                        "dTrackMomentum/F");
  poTrackTree_->Branch( "dTrackPt",       &oTrackVal_.dPt, 
                        "dTrackPt/F");
  poTrackTree_->Branch( "dTrackCharge",   &oTrackVal_.dCharge, 
                        "dTrackCharge/F");
  poTrackTree_->Branch( "dTrackEta",      &oTrackVal_.dEta, 
                        "dTrackEta/F");
  poTrackTree_->Branch( "dTrackPhi",      &oTrackVal_.dPhi, 
                        "dTrackPhi/F");
  poTrackTree_->Branch( "dTrackHits",     &oTrackVal_.dHits, 
                        "dTrackHits/F");
  poTrackTree_->Branch( "dTrackChi2",     &oTrackVal_.dChi2, 
                        "dTrackChi2/F");
  poTrackTree_->Branch( "dTrackChi2Norm", &oTrackVal_.dChi2Norm, 
                        "dTrackChi2Norm/F");
  poTrackTree_->Branch( "dTrackNdof",     &oTrackVal_.dNdof, 
                        "dTrackNdof/F");
}

void SliceTestNtupleMaker::endJob() {
  // Write buffers and Close opened file
  poOFile_->Write();
  poOFile_->Close();
}

void SliceTestNtupleMaker::analyze( const edm::Event      &roEVENT,
                                    const edm::EventSetup &roEVENT_SETUP) {

  // Extract basic parameters for trees
  oGenVal_.nRun      = roEVENT.id().run();
  oGenVal_.nLclEvent = roEVENT.id().event();
  oGenVal_.nLTime    = roEVENT.time().value();

  // Extract TrackerGeometry
  edm::ESHandle<TrackerGeometry> oHTrackerGeometry;
  try {
    LogDebug( "SliceTestNtupleMaker::analyze")
      << "\t* Extract Tracker Geometry";

    roEVENT_SETUP.get<TrackerDigiGeometryRecord>().get( oHTrackerGeometry);
  } catch( const cms::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::analyze") 
      << "Failed to extract Tracker Geometry. "
      << "Make sure you have included it in your CONFIG FILE";
    
    // Pass exception on
    throw;
  }

  // Extract Tracks
  edm::Handle<VRecoTracks> oVRecoTracks;
  try {
    LogDebug( "SliceTestNtupleMaker::analyze")
      << "\t* Extract Tracks";

    roEVENT.getByLabel( oConfig_.oITTrack, oVRecoTracks);
  } catch( const edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::analyze")
      << "Failed to Exctract Tracks. Make sure they are present in event and "
      << "specified InputTag for Tracks is correct";

    // Pass exception on
    throw;
  }

  // Extract Clusters
  edm::Handle<DSVSiStripClusters> oDSVClusters;
  try {
    LogDebug( "SliceTestNtupleMaker::analyze")
      << "\t* Extract clusters";

    roEVENT.getByLabel( oConfig_.oITCluster, oDSVClusters);
  } catch( const edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::analyze")
      << "Failed to Exctract SiStripClusters. Make sure they are present in event and "
      << "specified InputTag for Clusters is correct";

    // Pass exception on
    throw;
  }

  // Extract ClusterInfos
  edm::Handle<DSVSiStripClusterInfos> oDSVClusterInfos;
  try {
    LogDebug( "SliceTestNtupleMaker::analyze")
      << "\t* Extract cluster Infos";

    roEVENT.getByLabel( oConfig_.oITClusterInfo, oDSVClusterInfos);
  } catch( const edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::analyze")
      << "Failed to Exctract SiStripClusterInfos. Make sure they are present "
      << "in event and specified InputTag for ClusterInfos is correct";

    // Pass exception on
    throw;
  }

  // Extract Digis
  edm::Handle<extra::DSVSiStripDigis> oDSVDigis;
  try {
    LogDebug( "SliceTestNtupleMaker::analyze")
      << "\t* Extract Digis";

    roEVENT.getByLabel( oConfig_.oITDigi, oDSVDigis);
  } catch( const edm::Exception &roEX) {
    edm::LogError( "SliceTestNtupleMaker::analyze")
      << "Failed to Exctract SiStripDigis. Make sure they are present "
      << "in event and specified InputTag for Digis is correct";

    // Pass exception on
    throw;
  }

  // Construct association between Clusters and ClusterInfos
  buildClusterVsClusterInfoMap( oDSVClusters.product(),
                                oDSVClusterInfos.product());

  // Next variable will be used to mark clusters as those that belong to track
  std::set<const SiStripCluster *> oSProcessedClusters;

  // Initialize function that would extract UpdatedTSOSes for Tracks
  reco::GetUpdatedTSOSDebug getUpdatedTSOS( roEVENT_SETUP,
                                            oConfig_.oTTRHBuilder);

  int nTrackNum = 0;

  // Loop over Tracks
  for( VRecoTracks::const_iterator oTRACKS_ITER = oVRecoTracks->begin();
       oTRACKS_ITER != oVRecoTracks->end();
       ++oTRACKS_ITER) {
    
    // Get Hits<->UpdatedTSOS association for track: skip track if association
    // can not be gotten
    reco::MHitTSOS oMHitUpdatedTSOS;
    getUpdatedTSOS( *oTRACKS_ITER, oMHitUpdatedTSOS);

    if( oMHitUpdatedTSOS.empty()) { continue; }

    // Fill Track variable but first reset TrackVal
    oTrackVal_ = TrackVal();
    oTrackVal_.setTrack( *oTRACKS_ITER, ++nTrackNum);

    poTrackTree_->Fill();

    // Loop over Tracks Hits
    for( trackingRecHit_iterator oHitIter = oTRACKS_ITER->recHitsBegin();
         oHitIter != oTRACKS_ITER->recHitsEnd();
         ++oHitIter) {

      // Check if Hit is Valid otherwise skip it
      if( !( *oHitIter)->isValid()) { continue; }

      // Extract Hit Values but first reset HitVal
      oHitVal_ = HitVal();
      oHitVal_.setHit( ( *oHitIter).get(),
                       oHTrackerGeometry.product());

      // Extract Hit associated Cluster: first reset ClusterVal
      oClusterVal_ = ClusterVal();

      const TrackingRecHit *poREC_HIT = oHitIter->get();
      // In case of Mono or Stereo Hit dynamic_cast<...> will return non zero
      // value that is True
      const SiStripRecHit2D *poREC_HIT_2D = 
        dynamic_cast<const SiStripRecHit2D *>( poREC_HIT);

      // Check if Hit actually Projected one
      if( const ProjectedSiStripRecHit2D *poPROJECTED_REC_HIT_2D =
            dynamic_cast<const ProjectedSiStripRecHit2D *>( poREC_HIT)) {

        poREC_HIT_2D = &( poPROJECTED_REC_HIT_2D->originalHit());
      }

      if( poREC_HIT_2D) {
        // Regular hit: Mono or Stereo
        if( const SiStripCluster *poCLUSTER = 
              poREC_HIT_2D->cluster().operator->()) {

          oClusterVal_.setCluster( poCLUSTER, 
                                   oMClusterVsClusterInfo_, 
                                   oDSVDigis);

          // Mark Cluster as Processed
          oSProcessedClusters.insert( poCLUSTER);
        } else {
          // Cluster that is associated with Hit can not be extracted: do 
          // nothing - leave all cluster related values unchanged
          edm::LogError( "SliceTestNtupleMaker::analyze()")
            << "Cluster that is associated with hit can not be extracted";
        }
      } else if( const SiStripMatchedRecHit2D *poMATCHED_REC_HIT_2D =
                   dynamic_cast<const SiStripMatchedRecHit2D *>( poREC_HIT)) {

          oHitVal_.bMatched = true;

          if( const SiStripCluster *poMONO_CLUSTER = 
                poMATCHED_REC_HIT_2D->monoHit()->cluster().operator->()) {

            oClusterVal_.setCluster( poMONO_CLUSTER,
                                     oMClusterVsClusterInfo_,
                                     oDSVDigis);

            // Mark Cluster as Processed
            oSProcessedClusters.insert( poMONO_CLUSTER);

            poClusterTree_->Fill();

            // Reset Cluster Values
            oClusterVal_ = ClusterVal();
          } else {
            // Failed to extract Mono Cluster for Matched hit: do nothing
            // leave all cluster related values unchanged
            edm::LogError( "SliceTestNtupleMaker::analyze()")
              << "Failed to extract MONO Cluster for Matched hit";
          }

          if( const SiStripCluster *poSTEREO_CLUSTER =
                poMATCHED_REC_HIT_2D->stereoHit()->cluster().operator->()) {

            oClusterVal_.setCluster( poSTEREO_CLUSTER,
                                     oMClusterVsClusterInfo_,
                                     oDSVDigis);

            // Mark Cluster as Processed
            oSProcessedClusters.insert( poSTEREO_CLUSTER);
          } else {
            // Failed to extract Stereo Cluster for Matched hit: do nothing
            // leave all cluster related values unchanged
            edm::LogError( "SliceTestNtupleMaker::analyze()")
              << "Failed to extract STEREO Cluster for Matched hit";
          }
      } else {
        // Unsupported Hit: leave all cluster related values unchanged
        edm::LogError( "SliceTestNtupleMaker::analyze()")
          << "Unsupported Hit found";
      }

      poClusterTree_->Fill();
    } // End loop over hits
  } // End loop over tracks




  // +----------------------------------------------------
  // |  NON-TRACK CLUSTERS 
  // +----------------------------------------------------

  // Reset Track and Hit Variables
  oTrackVal_ = TrackVal();
  oHitVal_   = HitVal();

  // Loop over Cluster's collection: keys are DetId's
  for( DSVSiStripClusters::const_iterator oDSVITER = 
         oDSVClusters->begin();
       oDSVITER != oDSVClusters->end();
       ++oDSVITER) {

    // Loop over Clusters in given DetId
    const std::vector<SiStripCluster> &roVCLUSTERS = oDSVITER->data;
    for( std::vector<SiStripCluster>::const_iterator oVITER =
           roVCLUSTERS.begin();
         oVITER != roVCLUSTERS.end();
         ++oVITER) {

      // Skip all Track Clusters
      if( oSProcessedClusters.end() != oSProcessedClusters.find( &( *oVITER)) ) {
        continue;
      }

      // Extract Cluster parameters: first reset ClusterVal
      oClusterVal_ = ClusterVal();

      oClusterVal_.setCluster( &( *oVITER), 
                               oMClusterVsClusterInfo_,
                               oDSVDigis);

      poClusterTree_->Fill();
    } // End loop over Clusters in given DetId
  } // End loop over Clusters Collection
}

// +---------------------------------------------------------------------------
// |  PRIVATES 
// +---------------------------------------------------------------------------
/** 
* @brief 
*   Helper Class for Sorting ClusterVsClusterInfo map
*/
class ClusterClusterInfo_eq: 
  public std::unary_function<SiStripClusterInfo, bool> {

  public:
    explicit ClusterClusterInfo_eq( const SiStripCluster &roCLUSTER):
      nFIRST_STRIP( roCLUSTER.firstStrip()) {}

    bool operator()( const SiStripClusterInfo &roCLUSTER_INFO) const {
      return nFIRST_STRIP == roCLUSTER_INFO.firstStrip();
    }

  private:
    const int nFIRST_STRIP;
};

/** 
* @brief 
*   Method builds Cluster vs ClusterInfo association Map
* 
* @param poDSV_SI_STRIP_CLUSTERS       DetSetVector of Clusters 
* @param poDSV_SI_STRIP_CLUSTER_INFOS  DetSetVector of ClusterInfos
*/
void SliceTestNtupleMaker::buildClusterVsClusterInfoMap(
  const DSVSiStripClusters     *poDSV_SI_STRIP_CLUSTERS,
  const DSVSiStripClusterInfos *poDSV_SI_STRIP_CLUSTER_INFOS) {

  int nClusterMatchNotFound = 0;
  int nDetIdMatchNotFound   = 0;

  // Loop over Cluster's collection: keys are DetId's
  for( DSVSiStripClusters::const_iterator oDSV_ITER_CLUSTER = 
         poDSV_SI_STRIP_CLUSTERS->begin();
       oDSV_ITER_CLUSTER != poDSV_SI_STRIP_CLUSTERS->end();
       ++oDSV_ITER_CLUSTER) {
  
    // Find any ClusterInfos that match the same DetId
    DSVSiStripClusterInfos::const_iterator oDSV_ITER_CLUSTER_INFO = 
      poDSV_SI_STRIP_CLUSTER_INFOS->find( oDSV_ITER_CLUSTER->id);

    if( oDSV_ITER_CLUSTER_INFO != poDSV_SI_STRIP_CLUSTER_INFOS->end()) {

      const std::vector<SiStripCluster> &roVCLUSTERS = 
        oDSV_ITER_CLUSTER->data;

      const std::vector<SiStripClusterInfo> &roVCLUSTER_INFOS = 
        oDSV_ITER_CLUSTER_INFO->data;

      // Loop over Clusters in given DetId
      for( std::vector<SiStripCluster>::const_iterator oV_ITER_CLUSTER =
             roVCLUSTERS.begin();
           oV_ITER_CLUSTER != roVCLUSTERS.end();
           ++oV_ITER_CLUSTER) {

        // Now Try to find ClusterInfo that match given Cluster
        std::vector<SiStripClusterInfo>::const_iterator oV_ITER_CLUSTER_INFO =
          std::find_if( roVCLUSTER_INFOS.begin(),
                        roVCLUSTER_INFOS.end(),
                        ClusterClusterInfo_eq( *oV_ITER_CLUSTER));

        if( oV_ITER_CLUSTER_INFO != roVCLUSTER_INFOS.end()) {
          // Match found
          oMClusterVsClusterInfo_[&( *oV_ITER_CLUSTER)] = 
            &( *oV_ITER_CLUSTER_INFO);
        } else {
          ++nClusterMatchNotFound;
        }

      } // End loop over Clusters in given DetId
    } else {
      ++nDetIdMatchNotFound;
    }
  } // End loop over Clusters Collection

  if( nDetIdMatchNotFound) {
    LogDebug( "SliceTestNtupleMaker::buildClusterVsClusterInfoMap")
      << "\t * DetId   Match Not Found for " << nDetIdMatchNotFound;
  }

  if( nClusterMatchNotFound) {
    LogDebug( "SliceTestNtupleMaker::buildClusterVsClusterInfoMap")
      << "\t * Cluster Match Not Found for " << nClusterMatchNotFound;
  }
}

void SliceTestNtupleMaker::DetVal::setDet( const DetId &roDET_ID) {
  switch( roDET_ID.subdetId()) {
    case StripSubdetector::TIB:
      {
        TIBDetId oTIBDetId( roDET_ID.rawId());
        eSubDet       = StripSubdetector::TIB;
        nLayer        = oTIBDetId.layer();
        nString       = oTIBDetId.string()[2];
        nStringFwBw   = oTIBDetId.string()[0];
        nStringIntExt = oTIBDetId.string()[1];
        bStereo       = static_cast<bool>( oTIBDetId.stereo());
        break;
      }
    case StripSubdetector::TID:
      {
        TIDDetId oTIDDetId( roDET_ID.rawId());
        eSubDet     = StripSubdetector::TID;
        nWheel      = oTIDDetId.wheel();
        nSide       = oTIDDetId.side();
        nRing       = oTIDDetId.ring();
        nModuleFwBw = oTIDDetId.module()[0];
        bStereo     = static_cast<bool>( oTIDDetId.stereo());
        break;
      }
    case StripSubdetector::TOB:
      {
        TOBDetId oTOBDetId( roDET_ID.rawId());
        eSubDet  = StripSubdetector::TOB;
        nLayer   = oTOBDetId.layer();
        nRod     = oTOBDetId.rod()[1];
        nRodFwBw = oTOBDetId.rod()[0];
        bStereo  = static_cast<bool>( oTOBDetId.stereo());
        break;
      }
    case StripSubdetector::TEC:
      {
        TECDetId oTECDetId( roDET_ID.rawId());
        eSubDet    = StripSubdetector::TEC;
        nWheel     = oTECDetId.wheel();
        nRing      = oTECDetId.ring();
        nSide      = oTECDetId.side();
        nPetal     = oTECDetId.petal()[1];
        nPetalFwBw = oTECDetId.petal()[0];
        bStereo    = static_cast<bool>( oTECDetId.stereo());
        break;
      }
    default:
      // Currently unsupported Detector Id: reset DetVal
      break;
  }

  nModule = roDET_ID.rawId();
}

void SliceTestNtupleMaker::TrackVal::setTrack( 
  const reco::Track &roTRACK,
  const int         &rnTRACK_NUM) {

  bTrack    = true;
  nTrackNum = rnTRACK_NUM;
  dMomentum = roTRACK.p();
  dPt       = roTRACK.pt();
  dCharge   = roTRACK.charge();
  dEta      = roTRACK.eta();
  dPhi      = roTRACK.phi();
  dHits     = roTRACK.recHitsSize();
  dChi2     = roTRACK.chi2();
  dChi2Norm = roTRACK.normalizedChi2();
  dNdof     = roTRACK.ndof();
}

void SliceTestNtupleMaker::HitVal::setHit( 
  const TrackingRecHit  *poHIT,
  const TrackerGeometry *poTRACKER_GEOMETRY) {

  DetId oDetId( poHIT->geographicalId());
  oDetVal.setDet( oDetId);

  LocalPoint oLclPosition = poHIT->localPosition();
  oCoordsLocal.setPoint( oLclPosition);

  GlobalPoint oGlbPosition = 
    poTRACKER_GEOMETRY->idToDet( oDetId)->toGlobal( oLclPosition);
  oCoordsGlobal.setPoint( oGlbPosition);
}

void SliceTestNtupleMaker::ClusterVal::setCluster( 
  const SiStripCluster                      *poCLUSTER,
  const MClusterVsClusterInfo               &roDSVCLUSTER_INFOS,
  const edm::Handle<extra::DSVSiStripDigis> &roDSVDigis) {

  // Reset DetVal
  oDetVal     = DetVal();
  oDetVal.setDet( DetId( poCLUSTER->geographicalId()));

  nFirstStrip = poCLUSTER->firstStrip();
  dBaryCenter = poCLUSTER->barycenter();

  MClusterVsClusterInfo::const_iterator oC_VS_CI_ITER = 
    roDSVCLUSTER_INFOS.find( poCLUSTER);
  if( oC_VS_CI_ITER != roDSVCLUSTER_INFOS.end()) {

    // There is ClusterInfos match with Cluster
    dCharge    = oC_VS_CI_ITER->second->charge();
    dChargeL   = oC_VS_CI_ITER->second->chargeL();
    dChargeR   = oC_VS_CI_ITER->second->chargeR();
    dNoise     = oC_VS_CI_ITER->second->noise();
    dMaxCharge = oC_VS_CI_ITER->second->maxCharge();

    const std::vector<uint16_t> &roAMPLITUDES = 
      oC_VS_CI_ITER->second->stripAmplitudes();

    nWidth = roAMPLITUDES.size();
    dXTalk = extra::getSiStripClusterXTalk( *poCLUSTER,
                                            roDSVDigis);
  } else {
    // There is no association between Current Cluster and any ClusterInfo
    edm::LogError( "SliceTestNtupleMaker::ClusterVal::setCluster(...)")
      << "No assiciation available between Cluster and any ClusterInfo";
  }
}
