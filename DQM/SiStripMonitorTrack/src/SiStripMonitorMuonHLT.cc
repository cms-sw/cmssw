// -*- C++ -*-
//
// Package:    SiStripMonitorMuonHLT
// Class:      SiStripMonitorMuonHLT
// 
/**\class SiStripMonitorMuonHLT SiStripMonitorMuonHLT.cc DQM/SiStripMonitorMuonHLT/src/SiStripMonitorMuonHLT.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Eric Chabert
//         Created:  Wed Sep 23 17:26:42 CEST 2009
// $Id: SiStripMonitorMuonHLT.cc,v 1.15 2013/01/02 16:55:48 wmtan Exp $
//

#include "DQM/SiStripMonitorTrack/interface/SiStripMonitorMuonHLT.h"


//
// constructors and destructor
//
SiStripMonitorMuonHLT::SiStripMonitorMuonHLT (const edm::ParameterSet & iConfig)
{
  //now do what ever initialization is needed
  parameters_ = iConfig;
  verbose_ = parameters_.getUntrackedParameter<bool>("verbose",false);
  normalize_ = parameters_.getUntrackedParameter<bool>("normalize",true);
  printNormalize_ = parameters_.getUntrackedParameter<bool>("printNormalize",false);
  monitorName_ = parameters_.getUntrackedParameter<std::string>("monitorName","HLT/HLTMonMuon");
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt",-1);

  //booleans
  runOnClusters_ = parameters_.getUntrackedParameter<bool>("runOnClusters",true);
  runOnMuonCandidates_ = parameters_.getUntrackedParameter<bool>("runOnMuonCandidates",true);
  runOnTracks_ = parameters_.getUntrackedParameter<bool>("runOnTracks",true);

  //tags
  clusterCollectionTag_ = parameters_.getUntrackedParameter < edm::InputTag > ("clusterCollectionTag",edm::InputTag("hltSiStripRawToClustersFacility"));
  l3collectionTag_ = parameters_.getUntrackedParameter < edm::InputTag > ("l3MuonTag",edm::InputTag("hltL3MuonCandidates"));
  TrackCollectionTag_ = parameters_.getUntrackedParameter < edm::InputTag > ("trackCollectionTag",edm::InputTag("hltL3TkTracksFromL2"));
  //////////////////////////

  HistoNumber = 35;

  //services
  dbe_ = 0;
  if (!edm::Service < DQMStore > ().isAvailable ())
    {
      edm::LogError ("TkHistoMap") <<
	"\n------------------------------------------"
	"\nUnAvailable Service DQMStore: please insert in the configuration file an instance like" "\n\tprocess.load(\"DQMServices.Core.DQMStore_cfg\")" "\n------------------------------------------";
    }
  dbe_ = edm::Service < DQMStore > ().operator-> ();
  dbe_->setVerbose (0);

  tkdetmap_ = 0;
  if (!edm::Service < TkDetMap > ().isAvailable ())
    {
      edm::LogError ("TkHistoMap") <<
	"\n------------------------------------------"
	"\nUnAvailable Service TkHistoMap: please insert in the configuration file an instance like" "\n\tprocess.TkDetMap = cms.Service(\"TkDetMap\")" "\n------------------------------------------";
    }
  tkdetmap_ = edm::Service < TkDetMap > ().operator-> ();
  //////////////////////////

  outputFile_ = parameters_.getUntrackedParameter < std::string > ("outputFile","");
  if (outputFile_.size () != 0) edm::LogWarning ("HLTMuonDQMSource") << "Muon HLT Monitoring histograms will be saved to " << outputFile_ << std::endl;
  else outputFile_ = "MuonHLTDQM.root";

  bool disable = parameters_.getUntrackedParameter < bool > ("disableROOToutput",false);
  if (disable) outputFile_ = "";
  if (dbe_ != NULL) dbe_->setCurrentFolder (monitorName_);

}


SiStripMonitorMuonHLT::~SiStripMonitorMuonHLT ()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

float SiStripMonitorMuonHLT::GetEtaWeight(std::string label, GlobalPoint clustgp){
        float etaWeight = 1.;
	for (unsigned int i = 0; i < m_BinEta[label].size() - 1; i++){                      
        	if (m_BinEta[label][i] < clustgp.eta() && clustgp.eta() < m_BinEta[label][i+1]){
                	if (m_ModNormEta[label][i] > 0.1) etaWeight = 1./m_ModNormEta[label][i];
                	else etaWeight = 1.;
              	}       
        }
	return etaWeight; 
}

float SiStripMonitorMuonHLT::GetPhiWeight(std::string label, GlobalPoint clustgp){
        float phiWeight = 1.;
	for (unsigned int i = 0; i < m_BinPhi[label].size() - 1; i++){                      
        	if (m_BinPhi[label][i] < clustgp.phi() && clustgp.phi() < m_BinPhi[label][i+1]){
                	if (m_ModNormPhi[label][i] > 0.1) phiWeight = 1./m_ModNormPhi[label][i];
                	else phiWeight = 1.;
              	}       
        }
	return phiWeight; 
}

// ------------ method called to for each event  ------------
void
SiStripMonitorMuonHLT::analyze (const edm::Event & iEvent, const edm::EventSetup & iSetup)
{


#ifdef THIS_IS_AN_EVENT_EXAMPLE
  Handle < ExampleData > pIn;
  iEvent.getByLabel ("example", pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  ESHandle < SetupData > pSetup;
  iSetup.get < SetupRecord > ().get (pSetup);
#endif

  if (!dbe_)
    return;
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_ % prescaleEvt_ != 0)
    return;
  LogDebug ("SiStripMonitorHLTMuon") << " processing conterEvt_: " << counterEvt_ << std::endl;


  edm::ESHandle < TrackerGeometry > TG;
  iSetup.get < TrackerDigiGeometryRecord > ().get (TG);
  const TrackerGeometry *theTrackerGeometry = TG.product ();
  const TrackerGeometry & theTracker (*theTrackerGeometry);


  ///////////////////  Access to data   /////////////////////

  //Access to L3MuonCand
  edm::Handle < reco::RecoChargedCandidateCollection > l3mucands;
  bool accessToL3Muons = true;
  iEvent.getByLabel (l3collectionTag_, l3mucands);
  reco::RecoChargedCandidateCollection::const_iterator cand;

  //Access to clusters
  edm::Handle < edm::LazyGetter < SiStripCluster > >clusters;
  bool accessToClusters = true;
  iEvent.getByLabel (clusterCollectionTag_, clusters);
  edm::LazyGetter < SiStripCluster >::record_iterator clust;
 
  //Access to Tracks
  edm::Handle<reco::TrackCollection > trackCollection;
  bool accessToTracks = true;
  iEvent.getByLabel (TrackCollectionTag_, trackCollection);
  reco::TrackCollection::const_iterator track;
   /////////////////////////////////////////////////////


  if (runOnClusters_ && accessToClusters && !clusters.failedToGet () && clusters.isValid())
    {
      for (clust = clusters->begin_record (); clust != clusters->end_record (); ++clust)
	{
	  
	  uint detID = clust->geographicalId ();
	  std::stringstream ss;
	  int layer = tkdetmap_->FindLayer (detID);
	  std::string label = tkdetmap_->getLayerName (layer);
	  const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
	  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
	  // get the cluster position in local coordinates (cm) 
	  LocalPoint clustlp = topol->localPosition (clust->barycenter ());
	  GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
	  
	  //NORMALIZE HISTO IF ASKED
          float etaWeight = 1.;
          float phiWeight = 1.;
          if (normalize_){
	  	etaWeight = GetEtaWeight(label, clustgp);
	  	phiWeight = GetPhiWeight(label,clustgp);
          }        
          LayerMEMap[label.c_str ()].EtaDistribAllClustersMap->Fill (clustgp.eta (),etaWeight);
          LayerMEMap[label.c_str ()].PhiDistribAllClustersMap->Fill (clustgp.phi (),phiWeight);
          LayerMEMap[label.c_str ()].EtaPhiAllClustersMap->Fill (clustgp.eta (), clustgp.phi ());
	  tkmapAllClusters->add(detID,1.);
	}
    }

  if (runOnMuonCandidates_ && accessToL3Muons && !l3mucands.failedToGet () && l3mucands.isValid())
    {
      for (cand = l3mucands->begin (); cand != l3mucands->end (); ++cand)
	{
	  //TrackRef l3tk = cand->get < TrackRef > ();
	  const reco::Track* l3tk = cand->get < reco::TrackRef > ().get();
	  analyzeOnTrackClusters(l3tk, theTracker, true);	
	}			//loop over l3mucands
    }				//if l3seed
 
  if (runOnTracks_ && accessToTracks && !trackCollection.failedToGet() && trackCollection.isValid()){
	for (track = trackCollection->begin (); track != trackCollection->end() ; ++ track)
	  {
	    const reco::Track* tk =  &(*track);
	    analyzeOnTrackClusters(tk, theTracker, false);	
	  }
  }

}

void SiStripMonitorMuonHLT::analyzeOnTrackClusters( const reco::Track* l3tk, const TrackerGeometry & theTracker,  bool isL3MuTrack ){

	  for (size_t hit = 0; hit < l3tk->recHitsSize (); hit++)
	    {
	      //if hit is valid and in tracker say true
	      if (l3tk->recHit (hit)->isValid () == true && l3tk->recHit (hit)->geographicalId ().det () == DetId::Tracker)
		{
		  uint detID = l3tk->recHit (hit)->geographicalId ()();
		  
		  const SiStripRecHit1D *hit1D = dynamic_cast < const SiStripRecHit1D * >(l3tk->recHit (hit).get ());
		  const SiStripRecHit2D *hit2D = dynamic_cast < const SiStripRecHit2D * >(l3tk->recHit (hit).get ());
		  const SiStripMatchedRecHit2D *hitMatched2D = dynamic_cast < const SiStripMatchedRecHit2D * >(l3tk->recHit (hit).get ());
		  const ProjectedSiStripRecHit2D *hitProj2D = dynamic_cast < const ProjectedSiStripRecHit2D * >(l3tk->recHit (hit).get ());


		  // if SiStripRecHit1D
		  if (hit1D != 0)
		    {
		      if (hit1D->cluster_regional ().isNonnull ())
			{
			  if (hit1D->cluster_regional ().isAvailable ())
			    {
			      detID = hit1D->cluster_regional ()->geographicalId ();
			    }
			}
		      int layer = tkdetmap_->FindLayer (detID);
		      std::string label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hit1D->cluster_regional ()->barycenter ());
			      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
			      //NORMALIZE HISTO IF ASKED
			      float etaWeight = 1.;
          		      float phiWeight = 1.;
          		      if (normalize_){
	  			etaWeight = GetEtaWeight(label, clustgp);
	  			phiWeight = GetPhiWeight(label,clustgp);
   			      }        
			      if(!isL3MuTrack){
                              	LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapOnTrackClusters->add(detID,1.);
			      }
			      else{
                              	LayerMEMap[label.c_str ()].EtaDistribL3MuTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribL3MuTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiL3MuTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapL3MuTrackClusters->add(detID,1.);
			      }
			    }
			}
		    }
		  // if SiStripRecHit2D
		  if (hit2D != 0)
		    {
		      if (hit2D->cluster_regional ().isNonnull ())
			{
			  if (hit2D->cluster_regional ().isAvailable ())
			    {
			      detID = hit2D->cluster_regional ()->geographicalId ();
			    }
			}
		      int layer = tkdetmap_->FindLayer (detID);
		      std::string label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hit2D->cluster_regional ()->barycenter ());
			      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
	  		      
			      //NORMALIZE HISTO IF ASKED
			      float etaWeight = 1.;
          		      float phiWeight = 1.;
          		      if (normalize_){
	  			etaWeight = GetEtaWeight(label, clustgp);
	  			phiWeight = GetPhiWeight(label,clustgp);
   			      }
			      if(!isL3MuTrack){
                              	LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapOnTrackClusters->add(detID,1.);
			      }
			      else{
                              	LayerMEMap[label.c_str ()].EtaDistribL3MuTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribL3MuTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiL3MuTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapL3MuTrackClusters->add(detID,1.);
			      }
			    }
			}
		    }
		  // if SiStripMatchedRecHit2D  
		  if (hitMatched2D != 0)
		    {
		      //hit mono
	              detID = hitMatched2D->monoCluster().geographicalId ();
		      int layer = tkdetmap_->FindLayer (detID);
		      std::string label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hitMatched2D->monoCluster().barycenter ());
			      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
			      //NORMALIZE HISTO IF ASKED
			      float etaWeight = 1.;
          		      float phiWeight = 1.;
          		      if (normalize_){
	  			etaWeight = GetEtaWeight(label, clustgp);
	  			phiWeight = GetPhiWeight(label,clustgp);
   			      }        
			      if(!isL3MuTrack){
                              	LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapOnTrackClusters->add(detID,1.);
			      }
			      else{
                              	LayerMEMap[label.c_str ()].EtaDistribL3MuTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribL3MuTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiL3MuTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapL3MuTrackClusters->add(detID,1.);
			      }
			    }
			}

		      //hit stereo
	              detID = hitMatched2D->stereoCluster().geographicalId ();
		      layer = tkdetmap_->FindLayer (detID);
		      label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet2 = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet2 != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet2->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hitMatched2D->stereoCluster().barycenter ());
			      GlobalPoint clustgp = theGeomDet2->surface ().toGlobal (clustlp);
			      //NORMALIZE HISTO IF ASKED
			      float etaWeight = 1.;
          		      float phiWeight = 1.;
          		      if (normalize_){
	  			etaWeight = GetEtaWeight(label, clustgp);
	  			phiWeight = GetPhiWeight(label,clustgp);
   			      }        
			      if(!isL3MuTrack){
                              	LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapOnTrackClusters->add(detID,1.);
			      }
			      else{
                              	LayerMEMap[label.c_str ()].EtaDistribL3MuTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribL3MuTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiL3MuTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapL3MuTrackClusters->add(detID,1.);
			      }
			    }
			}

		    }

		  //if ProjectedSiStripRecHit2D
		  if (hitProj2D != 0)
		    {
		      if (hitProj2D->originalHit ().cluster_regional ().isNonnull ())
			{
			  if (hitProj2D->originalHit ().cluster_regional ().isAvailable ())
			    {
			      detID = hitProj2D->originalHit ().cluster_regional ()->geographicalId ();
			    }
			}
		      int layer = tkdetmap_->FindLayer (detID);
		      std::string label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hitProj2D->originalHit ().cluster_regional ()->barycenter ());
			      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
			      //NORMALIZE HISTO IF ASKED
			      float etaWeight = 1.;
          		      float phiWeight = 1.;
          		      if (normalize_){
	  			etaWeight = GetEtaWeight(label, clustgp);
	  			phiWeight = GetPhiWeight(label,clustgp);
   			      }        
			      if(!isL3MuTrack){
                              	LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapOnTrackClusters->add(detID,1.);
			      }
			      else{
                              	LayerMEMap[label.c_str ()].EtaDistribL3MuTrackClustersMap->Fill (clustgp.eta (),etaWeight);
                              	LayerMEMap[label.c_str ()].PhiDistribL3MuTrackClustersMap->Fill (clustgp.phi (),phiWeight);
                              	LayerMEMap[label.c_str ()].EtaPhiL3MuTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());  
	  			tkmapL3MuTrackClusters->add(detID,1.);
			      }
			    }
			}
		    }

		}
	    }			//loop over RecHits
}

void
SiStripMonitorMuonHLT::createMEs (const edm::EventSetup & es)
{

  // vector used 
  std::vector <float *> tgraphEta;
  std::vector <float *> tgraphPhi;
  std::vector <int> tgraphSize;

  std::vector <std::vector<float> > binningEta;
  std::vector <std::vector<float> > binningPhi;

  for (int p = 0; p < 34; p++){
    tgraphEta.push_back (new float[1000]);
    tgraphPhi.push_back (new float[1000]);    
  }

  // FOR COMPUTING BINNING
  std::map< std::string,std::vector<float> > m_BinEta_Prel ;
  std::map< std::string,std::vector<float> > m_PhiStripMod_Eta;
  std::map< std::string,std::vector<float> > m_PhiStripMod_Nb;
  
  //----------------

  //Get the tracker geometry
  edm::ESHandle < TrackerGeometry > TG;
  es.get < TrackerDigiGeometryRecord > ().get (TG);
  const TrackerGeometry *theTrackerGeometry = TG.product ();
  const TrackerGeometry & theTracker (*theTrackerGeometry);

  std::vector<DetId> Dets = theTracker.detUnitIds();  


  //CALL GEOMETRY METHOD
  GeometryFromTrackGeom(Dets,theTracker,es,m_PhiStripMod_Eta,m_PhiStripMod_Nb);


  ////////////////////////////////////////////////////
  ///  Creation of folder structure
  ///    and ME decleration
  ////////////////////////////////////////////////////

  std::string fullName, folder;

  //STRUCTURE OF DETECTORS
  int p =0;

  //Loop over layers
  for (int layer = 1; layer < HistoNumber; ++layer)
    {
      SiStripFolderOrganizer folderOrg;
      std::stringstream ss;
      SiStripDetId::SubDetector subDet;
      uint32_t subdetlayer, side;
      tkdetmap_->getSubDetLayerSide (layer, subDet, subdetlayer, side);
      folderOrg.getSubDetLayerFolderName (ss, subDet, subdetlayer, side);
      folder = ss.str ();
      dbe_->setCurrentFolder (monitorName_ + folder);

      LayerMEs layerMEs;
      layerMEs.EtaPhiAllClustersMap           = 0;
      layerMEs.EtaDistribAllClustersMap       = 0;  
      layerMEs.PhiDistribAllClustersMap       = 0;
      layerMEs.EtaPhiOnTrackClustersMap       = 0;
      layerMEs.EtaDistribOnTrackClustersMap   = 0;
      layerMEs.PhiDistribOnTrackClustersMap   = 0;  
      layerMEs.EtaPhiL3MuTrackClustersMap     = 0;
      layerMEs.EtaDistribL3MuTrackClustersMap = 0;
      layerMEs.PhiDistribL3MuTrackClustersMap = 0;

      std::string histoname;
      std::string title;
      std::string labelHisto = tkdetmap_->getLayerName (layer);

      std::string labelHisto_ID = labelHisto;
      labelHisto_ID.erase(3);

      //
      unsigned int sizePhi = 0;
      unsigned int sizeEta = 0;
      float * xbinsPhi = new float[100];
      float * xbinsEta = new float[100];

      //TEC && TID && TOB && TIB
      if (labelHisto_ID == "TEC" || labelHisto_ID == "TID" || labelHisto_ID == "TOB" || labelHisto_ID == "TIB"){

        // PHI BINNING
        //ADDING BORDERS
        m_BinPhi[labelHisto].push_back(-M_PI);
        m_BinPhi[labelHisto].push_back(M_PI);

        //SORTING
        sort(m_BinPhi[labelHisto].begin(),m_BinPhi[labelHisto].end());
        //CREATING XBIN VECTOR
        sizePhi = m_BinPhi[labelHisto].size();

        for (unsigned int i = 0; i < sizePhi; i++){
          xbinsPhi[i] = m_BinPhi[labelHisto][i];
        }

        //ETA BINNING
        std::vector <float > v_BinEta_Prel;
        // LOOPING ON RINGS
        for (unsigned int i = 0; i < 12; i++){
          // COMPUTE BARYCENTER IF NON NULL
          if (m_PhiStripMod_Nb[labelHisto][i] != 0 && fabs(m_PhiStripMod_Eta[labelHisto][i]) > 0.05){
            float EtaBarycenter = m_PhiStripMod_Eta[labelHisto][i]/m_PhiStripMod_Nb[labelHisto][i];
            v_BinEta_Prel.push_back(EtaBarycenter);
          }
        }

        //SORT THEM IN ETA
        sort(v_BinEta_Prel.begin(),v_BinEta_Prel.end());

        //RECOMPUTE THE BINS BY TAKING THE HALF OF THE DISTANCE
        for (unsigned int i = 0; i < v_BinEta_Prel.size(); i++){
          if (i == 0) m_BinEta[labelHisto].push_back(v_BinEta_Prel[i] - 0.15);
          if (i != 0) {
            float shift = v_BinEta_Prel[i] - v_BinEta_Prel[i-1];
            m_BinEta[labelHisto].push_back(v_BinEta_Prel[i] - shift/2.);
          }
          if (i == v_BinEta_Prel.size()-1) m_BinEta[labelHisto].push_back(v_BinEta_Prel[i] + 0.15);
        }

        sort(m_BinEta[labelHisto].begin(),m_BinEta[labelHisto].end());

        //CREATING XBIN VECTOR
        sizeEta = m_BinEta[labelHisto].size();

        for (unsigned int i = 0; i < sizeEta; i++){
          xbinsEta[i] = m_BinEta[labelHisto][i];
        }

      } // END SISTRIP DETECTORS

      // all clusters
      if(runOnClusters_){
      	histoname = "EtaAllClustersDistrib_" + labelHisto;
      	title = "#eta(All Clusters) in " + labelHisto;
      	layerMEs.EtaDistribAllClustersMap = dbe_->book1D (histoname, title, sizeEta - 1, xbinsEta);
      	histoname = "PhiAllClustersDistrib_" + labelHisto;
      	title = "#phi(All Clusters) in " + labelHisto;
      	layerMEs.PhiDistribAllClustersMap = dbe_->book1D (histoname, title, sizePhi - 1, xbinsPhi);
      	histoname = "EtaPhiAllClustersMap_" + labelHisto;
      	title = "#eta-#phi All Clusters map in " + labelHisto;
      	layerMEs.EtaPhiAllClustersMap = dbe_->book2D (histoname, title, sizeEta - 1, xbinsEta, sizePhi - 1, xbinsPhi);
      }
      // on track clusters
      if(runOnTracks_){
      	histoname = "EtaOnTrackClustersDistrib_" + labelHisto;
      	title = "#eta(OnTrack Clusters) in " + labelHisto;
      	layerMEs.EtaDistribOnTrackClustersMap = dbe_->book1D (histoname, title, sizeEta - 1, xbinsEta);
      	histoname = "PhiOnTrackClustersDistrib_" + labelHisto;
      	title = "#phi(OnTrack Clusters) in " + labelHisto;
      	layerMEs.PhiDistribOnTrackClustersMap = dbe_->book1D (histoname, title, sizePhi - 1, xbinsPhi);
      	histoname = "EtaPhiOnTrackClustersMap_" + labelHisto;
      	title = "#eta-#phi OnTrack Clusters map in " + labelHisto;
      	layerMEs.EtaPhiOnTrackClustersMap = dbe_->book2D (histoname, title, sizeEta - 1, xbinsEta, sizePhi - 1, xbinsPhi);
      }
      if(runOnMuonCandidates_){
      	// L3 muon track clusters
      	histoname = "EtaL3MuTrackClustersDistrib_" + labelHisto;
      	title = "#eta(L3MuTrack Clusters) in " + labelHisto;
      	layerMEs.EtaDistribL3MuTrackClustersMap = dbe_->book1D (histoname, title, sizeEta - 1, xbinsEta);
      	histoname = "PhiL3MuTrackClustersDistrib_" + labelHisto;
      	title = "#phi(L3MuTrack Clusters) in " + labelHisto;
      	layerMEs.PhiDistribL3MuTrackClustersMap = dbe_->book1D (histoname, title, sizePhi - 1, xbinsPhi);
      	histoname = "EtaPhiL3MuTrackClustersMap_" + labelHisto;
      	title = "#eta-#phi L3MuTrack Clusters map in " + labelHisto;
      	layerMEs.EtaPhiL3MuTrackClustersMap = dbe_->book2D (histoname, title, sizeEta - 1, xbinsEta, sizePhi - 1, xbinsPhi);
      }
      LayerMEMap[labelHisto] = layerMEs;

      //PUTTING ERRORS
      if(runOnClusters_){
      	LayerMEMap[labelHisto].EtaDistribAllClustersMap->getTH1F()->Sumw2();
     	LayerMEMap[labelHisto].PhiDistribAllClustersMap->getTH1F()->Sumw2();
      	LayerMEMap[labelHisto].EtaPhiAllClustersMap->getTH2F()->Sumw2();
      }
      if(runOnTracks_){
      	LayerMEMap[labelHisto].EtaDistribOnTrackClustersMap->getTH1F()->Sumw2();
      	LayerMEMap[labelHisto].PhiDistribOnTrackClustersMap->getTH1F()->Sumw2();
      	LayerMEMap[labelHisto].EtaPhiOnTrackClustersMap->getTH2F()->Sumw2();
      }
      if(runOnMuonCandidates_){
      	LayerMEMap[labelHisto].EtaDistribL3MuTrackClustersMap->getTH1F()->Sumw2();
      	LayerMEMap[labelHisto].PhiDistribL3MuTrackClustersMap->getTH1F()->Sumw2();
     	LayerMEMap[labelHisto].EtaPhiL3MuTrackClustersMap->getTH2F()->Sumw2();
      }
      
      p++;
    }   //end of loop over layers


  //CALL THE NORMALIZATION METHOD
  Normalizer(Dets,theTracker);

}				//end of method


void
SiStripMonitorMuonHLT::GeometryFromTrackGeom (std::vector<DetId> Dets,const TrackerGeometry & theTracker, const edm::EventSetup& es,
                                              std::map< std::string,std::vector<float> > & m_PhiStripMod_Eta,std::map< std::string,std::vector<float> > & m_PhiStripMod_Nb){

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  std::vector<std::string> v_LabelHisto;

  //Loop over DetIds
  //-----------------------------------------
  for(std::vector<DetId>::iterator detid_iterator =  Dets.begin(); detid_iterator!=Dets.end(); ++detid_iterator){
    uint32_t detid = (*detid_iterator)();

    if ( (*detid_iterator).null() == true) break;
    if (detid == 0)  break;

    // Select the propers detectors - avoid pixels
    const GeomDetUnit * GeomDet = theTracker.idToDetUnit(detid);
    const GeomDet::SubDetector detector = GeomDet->subDetector();

    int mylayer;
    std::string mylabelHisto;

    // SELECT SISTRIP DETECTORS
    if (detector == GeomDetEnumerators::TEC
        || detector == GeomDetEnumerators::TID
        || detector == GeomDetEnumerators::TOB
        || detector == GeomDetEnumerators::TIB
        ){

      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detid));
      const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));

      // Get the position of the 1st strip in local coordinates (cm) 
      LocalPoint clustlp = topol->localPosition (1.);
      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);

      // Get the eta, phi of modules
      mylayer = tkdetmap_->FindLayer (detid);
      mylabelHisto = tkdetmap_->getLayerName (mylayer);

      //      SiStripDetId stripdet = SiStripDetId(detid);

      // INITIALISATION OF m_PhiStripMod_Eta + BOOKING LAYERS

      //TEST IF NEW LAYER
      unsigned int count = 0;
      while (count < v_LabelHisto.size()){
        if (mylabelHisto == v_LabelHisto[count]) break;
        count++;
      }
      if (count == v_LabelHisto.size()){

        //FILL THE NEW LAYER
        v_LabelHisto.push_back(mylabelHisto);

        //INITIALIZE

        // LOOPING ON RINGS
        for (int i = 0; i < 12; i++){
          m_PhiStripMod_Eta[mylabelHisto].push_back(0.);
          m_PhiStripMod_Nb[mylabelHisto].push_back(0.);
        }
      }

      //TEC
      if (detector == GeomDetEnumerators::TEC ){

        

        //PHI BINNING
        //Select 7th ring
        if (tTopo->tecRing(detid) == 7){
          //SELECT FP
          if (tTopo->tecModule(detid) == 1 && tTopo->tecIsFrontPetal(detid) == true) m_BinPhi[mylabelHisto].push_back(clustgp.phi());
          //SELECT BP
          if (tTopo->tecModule(detid) == 1 && tTopo->tecIsBackPetal(detid) == true) m_BinPhi[mylabelHisto].push_back(clustgp.phi());
        }

        //ETA BINNING
        //Select arbitrary petal
        if (tTopo->tecPetalNumber(detid) == 1 ){
          m_PhiStripMod_Eta[mylabelHisto][tTopo->tecRing(detid)-1] = m_PhiStripMod_Eta[mylabelHisto][tTopo->tecRing(detid)-1] + clustgp.eta();
          m_PhiStripMod_Nb[mylabelHisto][tTopo->tecRing(detid)-1]++;
        }

      } //END TEC

      //TID
      if (detector == GeomDetEnumerators::TID ){

        

        //PHI BINNING
        //Select 1st ring
        if (tTopo->tecRing(detid) == 1){
          //SELECT MONO
          if (tTopo->tecIsFrontPetal(detid) == true && tTopo->tecIsStereo(detid) == false) m_BinPhi[mylabelHisto].push_back(clustgp.phi());
          //SELECT STEREO
          if (tTopo->tecIsFrontPetal(detid) == true && tTopo->tecIsStereo(detid) == true) m_BinPhi[mylabelHisto].push_back(clustgp.phi());
        }

        //ETA BINNING
        //Select arbitrary line in eta (phi fixed)
        if (tTopo->tecModule(detid) == 1){
          m_PhiStripMod_Eta[mylabelHisto][tTopo->tecRing(detid)-1] = m_PhiStripMod_Eta[mylabelHisto][tTopo->tecRing(detid)-1] + clustgp.eta();
          m_PhiStripMod_Nb[mylabelHisto][tTopo->tecRing(detid)-1]++;
        }

      } //END TID

      //TOB
      if (detector == GeomDetEnumerators::TOB ){

        
        //PHI BINNING
        //Select arbitrary line in phi (detid)ta fixed)
        if (tTopo->tecModule(detid) == 1 && tTopo->tecIsZMinusSide(detid) == true){
          //SELECT MONO
          if (tTopo->tecIsStereo(detid) == false) m_BinPhi[mylabelHisto].push_back(clustgp.phi());
        }

        //ETA BINNING
        //Select arbitrary rod
        if ( (tTopo->tobRod(detid) == 2 && tTopo->tobIsStereo(detid) == false)
             || (tTopo->tobRod(detid) == 1 && tTopo->tobIsStereo(detid) == true)
             ){
          if (tTopo->tobIsZMinusSide(detid) == true){
            m_PhiStripMod_Eta[mylabelHisto][tTopo->tobModule(detid)-1] = m_PhiStripMod_Eta[mylabelHisto][tTopo->tobModule(detid)-1] + clustgp.eta();
            m_PhiStripMod_Nb[mylabelHisto][tTopo->tobModule(detid)-1]++;
          }
          if (tTopo->tobIsZMinusSide(detid) == false){
            m_PhiStripMod_Eta[mylabelHisto][tTopo->tobModule(detid)+5] = m_PhiStripMod_Eta[mylabelHisto][tTopo->tobModule(detid)+5] + clustgp.eta();
            m_PhiStripMod_Nb[mylabelHisto][tTopo->tobModule(detid)+5]++;
          }
        }

      } //END TOB

      //TIB
      if (detector == GeomDetEnumerators::TIB ){

        

        //PHI BINNING
        //Select arbitrary line in phi (eta fixed)
        if (tTopo->tibModule(detid) == 1 && tTopo->tibIsZMinusSide(detid) == true){
          //SELECT MONO
          if (tTopo->tibIsInternalString(detid) == true && tTopo->tibIsStereo(detid) == false) m_BinPhi[mylabelHisto].push_back(clustgp.phi());
        }

        //ETA BINNING
        //Select arbitrary string
        if ( (tTopo->tibString(detid) == 2 && tTopo->tibIsStereo(detid) == false)
             || (tTopo->tibString(detid) == 1 && tTopo->tibIsStereo(detid) == true)
             ){
          if (tTopo->tibIsZMinusSide(detid) == true){
            if (tTopo->tibIsInternalString(detid) == true){
              m_PhiStripMod_Eta[mylabelHisto][tTopo->tibModule(detid)-1] = m_PhiStripMod_Eta[mylabelHisto][tTopo->tibModule(detid)-1] + clustgp.eta();
              m_PhiStripMod_Nb[mylabelHisto][tTopo->tibModule(detid)-1]++;
            }
            if (tTopo->tibIsInternalString(detid) == false){
              m_PhiStripMod_Eta[mylabelHisto][tTopo->tibModule(detid)+2] = m_PhiStripMod_Eta[mylabelHisto][tTopo->tibModule(detid)+2] + clustgp.eta();
              m_PhiStripMod_Nb[mylabelHisto][tTopo->tibModule(detid)+2]++;
            }
          }
          if (tTopo->tibIsZMinusSide(detid) == false){
            if (tTopo->tibIsInternalString(detid) == true){
              m_PhiStripMod_Eta[mylabelHisto][tTopo->tibModule(detid)+5] = m_PhiStripMod_Eta[mylabelHisto][tTopo->tibModule(detid)+5] + clustgp.eta();
              m_PhiStripMod_Nb[mylabelHisto][tTopo->tibModule(detid)+5]++;
            }
            if (tTopo->tibIsInternalString(detid) == false){
              m_PhiStripMod_Eta[mylabelHisto][tTopo->tibModule(detid)+8] = m_PhiStripMod_Eta[mylabelHisto][tTopo->tibModule(detid)+8] + clustgp.eta();
              m_PhiStripMod_Nb[mylabelHisto][tTopo->tibModule(detid)+8]++;
            }
          }
        }

      } //END TIB

    } // END SISTRIP DETECTORS
  } // END DETID LOOP

} //END OF METHOD



void
SiStripMonitorMuonHLT::Normalizer (std::vector<DetId> Dets,const TrackerGeometry & theTracker){
  
  
  std::vector<std::string> v_LabelHisto;

  //Loop over DetIds
  //-----------------------------------------
  for(std::vector<DetId>::iterator detid_iterator =  Dets.begin(); detid_iterator!=Dets.end(); detid_iterator++){
    uint32_t detid = (*detid_iterator)();
    
    if ( (*detid_iterator).null() == true) break;
    if (detid == 0)  break;  
    
    // Select the propers detectors - avoid pixels
    const GeomDetUnit * GeomDet = theTracker.idToDetUnit(detid);
    const GeomDet::SubDetector detector = GeomDet->subDetector();

    int mylayer;
    std::string mylabelHisto;
    
    // SELECT SISTRIP DETECTORS
    if (detector == GeomDetEnumerators::TEC 
        || detector == GeomDetEnumerators::TID
        || detector == GeomDetEnumerators::TOB
        || detector == GeomDetEnumerators::TIB
        ){
      
      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detid));
      //      const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));

      // Get the eta, phi of modules
      mylayer = tkdetmap_->FindLayer (detid);
      mylabelHisto = tkdetmap_->getLayerName (mylayer);

      //      SiStripDetId stripdet = SiStripDetId(detid);

      // INITIALISATION OF m_ModNormEta + BOOKING LAYERS

      //TEST IF NEW LAYER
      unsigned int count = 0;

      while (count < v_LabelHisto.size()){
        if (mylabelHisto == v_LabelHisto[count]) break;
        count++;
      }

      if (count == v_LabelHisto.size()){
        //FILL THE NEW LAYER
        v_LabelHisto.push_back(mylabelHisto);

        //INITIALIZE    
        // LOOPING ON ETA VECTOR
        for (unsigned int i = 0; i < m_BinEta[mylabelHisto].size() -1; i++){
          m_ModNormEta[mylabelHisto].push_back(0.);
        }

        // LOOPING ON PHI VECTOR
        for (unsigned int i = 0; i < m_BinPhi[mylabelHisto].size() -1; i++){
          m_ModNormPhi[mylabelHisto].push_back(0.);
        }
      }

      // Get the position of the 1st strip in local coordinates (cm) 
      //      LocalPoint clustlp_1 = topol->localPosition (1.);
      //      GlobalPoint clustgp_1 = theGeomDet->surface ().toGlobal (clustlp_1);

      // Get the position of the center of the module
      LocalPoint clustlp(0.,0.);
      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);

      // Get the position of the last strip
      //      LocalPoint Border_clustlp = topol->localPosition (topol->nstrips());
      //      GlobalPoint Border_clustgp = theGeomDet->surface ().toGlobal (Border_clustlp);

      //GETTING SURFACE VALUE
      const BoundPlane& GeomDetSurface = GeomDet->surface();
      const Bounds& bound = GeomDetSurface.bounds();        
                                                    
      std::string labelHisto_ID = mylabelHisto;
      labelHisto_ID.erase(3);             
                             
      float length = 0.;
      float width = 0.; 

      std::vector <GlobalPoint> v_Edge_G;
                                    
      float ratio = 0.;
      float factor = 1.;
      
      //RECTANGULAR BOUNDS
      if (labelHisto_ID == "TOB" || labelHisto_ID == "TIB"){
        const RectangularPlaneBounds *rectangularBound = dynamic_cast < const RectangularPlaneBounds * >(& bound);                                                  
        length = rectangularBound->length();
        width = rectangularBound->width();                                                                                                                        
        ratio = width/length;
            
        //EDGES POINTS
        LocalPoint topleft(-width/2., length/2.);
        LocalPoint topright(width/2., length/2.);                                                                                                                   LocalPoint botleft(-width/2., -length/2.);
        LocalPoint botright(width/2., -length/2.);                                                                                                                                  
        v_Edge_G.push_back(theGeomDet->surface ().toGlobal (topleft));
        v_Edge_G.push_back(theGeomDet->surface ().toGlobal (topright));
        v_Edge_G.push_back(theGeomDet->surface ().toGlobal (botleft));                                                                                              v_Edge_G.push_back(theGeomDet->surface ().toGlobal (botright));
      }                                                                                                                                                            
      //TRAPEZOIDAL BOUNDS
      if (labelHisto_ID == "TEC" || labelHisto_ID == "TID"){    
        const TrapezoidalPlaneBounds *trapezoidalBound = dynamic_cast < const TrapezoidalPlaneBounds * >(& bound);

        length = trapezoidalBound->length();
        width = trapezoidalBound->widthAtHalfLength();

        ratio = width/length;

        //EDGES POINTS
        LocalPoint topleft(-width/2., length/2.);
        LocalPoint topright(width/2., length/2.);
        LocalPoint botleft(-width/2., -length/2.);
        LocalPoint botright(width/2., -length/2.);

        v_Edge_G.push_back(theGeomDet->surface ().toGlobal (topleft));
        v_Edge_G.push_back(theGeomDet->surface ().toGlobal (topright));
        v_Edge_G.push_back(theGeomDet->surface ().toGlobal (botleft));
        v_Edge_G.push_back(theGeomDet->surface ().toGlobal (botright));
      }

      //SORTING EDGES POINTS
      GlobalPoint top_left_G;
      GlobalPoint top_rightG;
      GlobalPoint bot_left_G;
      GlobalPoint bot_rightG;

      std::vector <bool> v_Fill;
      v_Fill.push_back(false);
      v_Fill.push_back(false);
      v_Fill.push_back(false);
      v_Fill.push_back(false);

      for (unsigned int i =0 ; i< v_Edge_G.size() ; i++){
        if (v_Edge_G[i].eta() < clustgp.eta()){
          if (v_Edge_G[i].phi() < clustgp.phi()) {
            bot_left_G = v_Edge_G[i];
            v_Fill[0] = true;
          }
          if (v_Edge_G[i].phi() > clustgp.phi()){
            top_left_G = v_Edge_G[i];
            v_Fill[1] = true;
          }
        }
        if (v_Edge_G[i].eta() > clustgp.eta()){
          if (v_Edge_G[i].phi() < clustgp.phi()){
            bot_rightG = v_Edge_G[i];
            v_Fill[2] = true;
          }
          if (v_Edge_G[i].phi() > clustgp.phi()){
            top_rightG = v_Edge_G[i];
            v_Fill[3] = true;
          }
        }
      }

      //USE EDGES FOR COMPUTING WIDTH AND LENGTH

      float G_length = 0.;
      float G_width = 0.;

      bool flag_border = false;

      if (v_Fill[0] == true
          && v_Fill[1] == true
          && v_Fill[2] == true
          && v_Fill[3] == true){

        //LENGTH BETWEEN TL AND TR
        G_length = sqrt( (top_left_G.x()-top_rightG.x())*(top_left_G.x()-top_rightG.x()) + (top_left_G.y()-top_rightG.y())*(top_left_G.y()-top_rightG.y()) + (top_left_G.z()-top_rightG.z())*(top_left_G.z()-top_rightG.z()) );

        //WIDTH BETWEEN BL AND TL
        G_width = sqrt( (bot_left_G.x()-top_left_G.x())*(bot_left_G.x()-top_left_G.x()) + (bot_left_G.y()-top_left_G.y())*(bot_left_G.y()-top_left_G.y()) + (bot_left_G.z()-top_left_G.z())*(bot_left_G.z()-top_left_G.z()) );

      }
      else {

        // MODULE IN THE PHI BORDER (-PI,PI)
        flag_border = true;

        //SORT THE EDGES POINTS 
        for (unsigned int i =0 ; i< v_Edge_G.size() ; i++){

          if (v_Edge_G[i].phi() > 0. ){
            if (v_Edge_G[i].eta() < clustgp.eta()){
              bot_left_G = v_Edge_G[i];
            }
            if (v_Edge_G[i].eta() > clustgp.eta()){
              bot_rightG = v_Edge_G[i];
            }
          }
          if (v_Edge_G[i].phi() < 0. ){
            if (v_Edge_G[i].eta() < clustgp.eta()){
              top_left_G = v_Edge_G[i];
            }
            if (v_Edge_G[i].eta() > clustgp.eta()){
              top_rightG = v_Edge_G[i];
            }
          }
        }

        // XYZ WIDTH AND LENGTH
        G_length = sqrt( (top_left_G.x()-top_rightG.x())*(top_left_G.x()-top_rightG.x()) + (top_left_G.y()-top_rightG.y())*(top_left_G.y()-top_rightG.y()) + (top_left_G.z()-top_rightG.z())*(top_left_G.z()-top_rightG.z()) );
        G_width = G_length*ratio;
      }


      //ETA PLOTS
      //unsigned int LastBinEta = m_BinEta[mylabelHisto].size() - 2;
      for (unsigned int i = 0; i < m_BinEta[mylabelHisto].size() - 1; i++){
        if (m_BinEta[mylabelHisto][i] <= clustgp.eta() && clustgp.eta() < m_BinEta[mylabelHisto][i+1]){

          // NO NEED TO DO CORRECTIONS FOR ETA
          m_ModNormEta[mylabelHisto][i] = m_ModNormEta[mylabelHisto][i] + factor*G_length*G_width;

        }
      } //END ETA

      //PHI PLOTS
      unsigned int LastBinPhi = m_BinPhi[mylabelHisto].size() - 2;
      for (unsigned int i = 0; i < m_BinPhi[mylabelHisto].size() - 1; i++){
        if (m_BinPhi[mylabelHisto][i] <= clustgp.phi() && clustgp.phi() < m_BinPhi[mylabelHisto][i+1]){

          // SCRIPT TO INTEGRATE THE SURFACE INTO PHI BIN

          float phiMin = std::min(bot_left_G.phi(),bot_rightG.phi());
          float phiMax = std::max(top_left_G.phi(),top_rightG.phi());

          bool offlimit_prev = false;
          bool offlimit_foll = false;

          if (phiMin < m_BinPhi[mylabelHisto][i]) offlimit_prev = true;
          if (i != LastBinPhi){
            if (phiMax > m_BinPhi[mylabelHisto][i+1]) offlimit_foll = true;
          }

          //LOOKING FOR THE INTERSECTION POINTS   
          float MidPoint_X_prev;
          float MidPoint_Y_prev;
          float MidPoint_Z_prev;
          float MidPoint_X_foll;
          float MidPoint_Y_foll;
          float MidPoint_Z_foll;

          // OFF LIMIT IN THE PREVIOUS BIN
          if (offlimit_prev){

            // BL TL
            float tStar1 = (m_BinPhi[mylabelHisto][i]-bot_left_G.phi())/(top_left_G.phi()-bot_left_G.phi());

            // BR TR
            float tStar2 = (m_BinPhi[mylabelHisto][i]-bot_rightG.phi())/(top_rightG.phi()-bot_rightG.phi());

            if (tStar1 < 0.) tStar1 = 0.;
            if (tStar2 < 0.) tStar2 = 0.;

            //FIND Z OF STAR POINT
            float xStar1 = bot_left_G.x() + (tStar1*1.)*(top_left_G.x()-bot_left_G.x());
            float xStar2 = bot_rightG.x() + (tStar2*1.)*(top_rightG.x()-bot_rightG.x());

            float yStar1 = bot_left_G.y() + (tStar1*1.)*(top_left_G.y()-bot_left_G.y());
            float yStar2 = bot_rightG.y() + (tStar2*1.)*(top_rightG.y()-bot_rightG.y());

            float zStar1 = bot_left_G.z() + (tStar1*1.)*(top_left_G.z()-bot_left_G.z());
            float zStar2 = bot_rightG.z() + (tStar2*1.)*(top_rightG.z()-bot_rightG.z());

            //MIDPOINT
            MidPoint_X_prev = (xStar1 + xStar2)/2.;
            MidPoint_Y_prev = (yStar1 + yStar2)/2.;
            MidPoint_Z_prev = (zStar1 + zStar2)/2.;
          }

          if (offlimit_prev == false){
            MidPoint_X_prev = (bot_left_G.x() + bot_rightG.x())/2.;
            MidPoint_Y_prev = (bot_left_G.y() + bot_rightG.y())/2.;
            MidPoint_Z_prev = (bot_left_G.z() + bot_rightG.z())/2.;
          }

          // OFF LIMIT IN THE FOLLOWING BIN
          if (offlimit_foll){

             // BL TL
            float tStar1 = (m_BinPhi[mylabelHisto][i+1]-bot_left_G.phi())/(top_left_G.phi()-bot_left_G.phi());

            // BR TR
            float tStar2 = (m_BinPhi[mylabelHisto][i+1]-bot_rightG.phi())/(top_rightG.phi()-bot_rightG.phi());

            if (tStar1 > 1.) tStar1 = 1.;
            if (tStar2 > 1.) tStar2 = 1.;

            //FIND Z OF STAR POINT                  
            float xStar1 = bot_left_G.x() + (tStar1*1.)*(top_left_G.x()-bot_left_G.x());
            float xStar2 = bot_rightG.x() + (tStar2*1.)*(top_rightG.x()-bot_rightG.x());

            float yStar1 = bot_left_G.y() + (tStar1*1.)*(top_left_G.y()-bot_left_G.y());
            float yStar2 = bot_rightG.y() + (tStar2*1.)*(top_rightG.y()-bot_rightG.y());

            float zStar1 = bot_left_G.z() + (tStar1*1.)*(top_left_G.z()-bot_left_G.z());
            float zStar2 = bot_rightG.z() + (tStar2*1.)*(top_rightG.z()-bot_rightG.z());

            //MIDPOINT
            MidPoint_X_foll = (xStar1 + xStar2)/2.;
            MidPoint_Y_foll = (yStar1 + yStar2)/2.;
            MidPoint_Z_foll = (zStar1 + zStar2)/2.;
          }

          if (offlimit_foll == false){
            MidPoint_X_foll = (top_left_G.x() + top_rightG.x())/2.;
            MidPoint_Y_foll = (top_left_G.y() + top_rightG.y())/2.;
            MidPoint_Z_foll = (top_left_G.z() + top_rightG.z())/2.;
          }

          //COMPUTE THE B AND T EDGES 
          float EdgePoint_X_B = (bot_left_G.x() + bot_rightG.x())/2.;
          float EdgePoint_Y_B = (bot_left_G.y() + bot_rightG.y())/2.;
          float EdgePoint_Z_B = (bot_left_G.z() + bot_rightG.z())/2.;

          float EdgePoint_X_T = (top_left_G.x() + top_rightG.x())/2.;
          float EdgePoint_Y_T = (top_left_G.y() + top_rightG.y())/2.;
          float EdgePoint_Z_T = (top_left_G.z() + top_rightG.z())/2.;
          // FILL INSIDE WIDTH
          float G_width_Ins = sqrt( (MidPoint_X_foll-MidPoint_X_prev)*(MidPoint_X_foll-MidPoint_X_prev) + (MidPoint_Y_foll-MidPoint_Y_prev)*(MidPoint_Y_foll-MidPoint_Y_prev) + (MidPoint_Z_foll-MidPoint_Z_prev)*(MidPoint_Z_foll-MidPoint_Z_prev) );

          //IF BORDER
          if (flag_border){

            // A) 3 POINT AND 1 POINT
            if (i != 0 && i != LastBinPhi){
              m_ModNormPhi[mylabelHisto][i] = m_ModNormPhi[mylabelHisto][i] + factor*G_length*G_width;
            }

            // B) MODULE SPLITTED IN TWO
            if (i == 0 || i == LastBinPhi){
              float PhiBalance = 0.;
              if (clustgp.phi() > 0.) PhiBalance = clustgp.phi() - M_PI ;
              if (clustgp.phi() < 0.) PhiBalance = clustgp.phi() + M_PI ;

              // Average Phi width of a phi bin
              float Phi_Width = m_BinPhi[mylabelHisto][3] - m_BinPhi[mylabelHisto][2];

              float weight_FirstBin = (1.+ (PhiBalance/(Phi_Width/2.)))/2. ;
              float weight_LastBin = fabs(1. - weight_FirstBin);

              m_ModNormPhi[mylabelHisto][0] = m_ModNormPhi[mylabelHisto][0] + weight_FirstBin*(factor*G_length*G_width);
              m_ModNormPhi[mylabelHisto][LastBinPhi] = m_ModNormPhi[mylabelHisto][LastBinPhi] + weight_LastBin*(factor*G_length*G_width);
            }
          }

          if (flag_border == false){

            // A) SURFACE TOTALY CONTAINED IN THE BIN
            if (offlimit_prev == false && offlimit_foll == false){
              m_ModNormPhi[mylabelHisto][i] = m_ModNormPhi[mylabelHisto][i] + factor*G_length*G_width;
            }

            // B) SURFACE CONTAINED IN 2 BINS
            if ((offlimit_prev == true && offlimit_foll == false)
                ||(offlimit_prev == false && offlimit_foll == true) ){
              float G_width_Out = fabs(G_width - G_width_Ins);

              //FILL INSIDE CELL            
              m_ModNormPhi[mylabelHisto][i] = m_ModNormPhi[mylabelHisto][i] + factor*G_width_Ins*G_length;

              //FILL OFF LIMITS CELLS
              if (offlimit_prev && i != 0) m_ModNormPhi[mylabelHisto][i-1] = m_ModNormPhi[mylabelHisto][i-1] + factor*G_width_Out*G_length;
              if (offlimit_foll && i != LastBinPhi) m_ModNormPhi[mylabelHisto][i+1] = m_ModNormPhi[mylabelHisto][i+1] + factor*G_width_Out*G_length;
            }

            // C) SURFACE CONTAINED IN 3 BINS
            if (offlimit_prev == true && offlimit_foll == true){

              //COMPUTE OFF LIMITS LENGTHS
              float G_width_T =  sqrt( (MidPoint_X_foll-EdgePoint_X_T)*(MidPoint_X_foll-EdgePoint_X_T) + (MidPoint_Y_foll-EdgePoint_Y_T)*(MidPoint_Y_foll-EdgePoint_Y_T) + (MidPoint_Z_foll-EdgePoint_Z_T)*(MidPoint_Z_foll-EdgePoint_Z_T) );
              float G_width_B =  sqrt( (MidPoint_X_prev-EdgePoint_X_B)*(MidPoint_X_prev-EdgePoint_X_B) + (MidPoint_Y_prev-EdgePoint_Y_B)*(MidPoint_Y_prev-EdgePoint_Y_B) + (MidPoint_Z_prev-EdgePoint_Z_B)*(MidPoint_Z_prev-EdgePoint_Z_B) );

              //FOR SAFETY
              if (i != 0 && i != LastBinPhi){
                //FILL INSIDE CELL          
                m_ModNormPhi[mylabelHisto][i] = m_ModNormPhi[mylabelHisto][i] + factor*G_width_Ins*G_length;

                //FILL OFF LIMITS CELLS
                if (i != 0) m_ModNormPhi[mylabelHisto][i-1] = m_ModNormPhi[mylabelHisto][i-1] + factor*G_width_B*G_length;
                if (i != LastBinPhi) m_ModNormPhi[mylabelHisto][i+1] = m_ModNormPhi[mylabelHisto][i+1] + factor*G_width_T*G_length;
              }

            }
          }
        }
      } // END PHI

    } // END SISTRIP DETECTORS

  } // END DETID LOOP

  //PRINT NORMALIZATION IF ASKED
  if (printNormalize_) {
    TFile output("MuonHLTDQMNormalization.root","recreate");
    output.cd();
    PrintNormalization(v_LabelHisto);
    output.Close();
  }

} //END METHOD



void
SiStripMonitorMuonHLT::PrintNormalization (std::vector<std::string> v_LabelHisto)
{
  std::vector <TH1F *> h_ModNorm_Eta;
  std::vector <TH1F *> h_ModNorm_Phi;

  for (unsigned int p = 0; p < v_LabelHisto.size(); p++){
   
    std::string titleHistoEta = v_LabelHisto[p] + "_eta" ;    
    std::string titleHistoPhi = v_LabelHisto[p] + "_phi" ;  

    std::string labelHisto = v_LabelHisto[p];
    
    float * xbinsPhi = new float[100];
    float * xbinsEta = new float[100];

    //CREATING XBIN VECTOR
    unsigned int sizePhi = m_BinPhi[labelHisto].size();
    for (unsigned int i = 0; i < sizePhi; i++){
      xbinsPhi[i] = m_BinPhi[labelHisto][i];
    }
    //CREATING XBIN VECTOR
    unsigned int sizeEta = m_BinEta[labelHisto].size();
    for (unsigned int i = 0; i < sizeEta; i++){
      xbinsEta[i] = m_BinEta[labelHisto][i];
    }
       
    h_ModNorm_Eta.push_back(new TH1F (titleHistoEta.c_str(),titleHistoEta.c_str(),sizeEta - 1,xbinsEta));
    h_ModNorm_Phi.push_back(new TH1F (titleHistoPhi.c_str(),titleHistoPhi.c_str(),sizePhi - 1,xbinsPhi));
    
    for (unsigned int i = 0; i < m_ModNormEta[labelHisto].size(); i++){
      (*h_ModNorm_Eta[p]).SetBinContent(i+1,m_ModNormEta[labelHisto][i]);
    }
    for (unsigned int i = 0; i < m_ModNormPhi[labelHisto].size(); i++){
      (*h_ModNorm_Phi[p]).SetBinContent(i+1,m_ModNormPhi[labelHisto][i]);
    }

    (*h_ModNorm_Eta[p]).Write();
    (*h_ModNorm_Phi[p]).Write();
  }
    
} 


// ------------ method called once each job just before starting event loop  ------------
void
SiStripMonitorMuonHLT::beginRun (const edm::Run& run, const edm::EventSetup & es)
{
  if (dbe_)
    {
      if (monitorName_ != "")
	monitorName_ = monitorName_ + "/";
      edm::LogInfo ("HLTMuonDQMSource") << "===>DQM event prescale = " << prescaleEvt_ << " events " << std::endl;
      createMEs (es);
      //create TKHistoMap
      if(runOnClusters_)
      	tkmapAllClusters = new TkHistoMap("HLT/HLTMonMuon/SiStrip" ,"TkHMap_AllClusters",0.0,0);
      if(runOnTracks_)
      	tkmapOnTrackClusters = new TkHistoMap("HLT/HLTMonMuon/SiStrip" ,"TkHMap_OnTrackClusters",0.0,0);
      if(runOnMuonCandidates_)
      	tkmapL3MuTrackClusters = new TkHistoMap("HLT/HLTMonMuon/SiStrip" ,"TkHMap_L3MuTrackClusters",0.0,0);
    }
}

// ------------ method called once each job just after ending the event loop  ------------
void
SiStripMonitorMuonHLT::endJob ()
{
  edm::LogInfo ("SiStripMonitorHLTMuon") << "analyzed " << counterEvt_ << " events";
  return;
}

DEFINE_FWK_MODULE(SiStripMonitorMuonHLT);
