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
// $Id: SiStripMonitorMuonHLT.cc,v 1.1 2009/10/05 17:05:48 echabert Exp $
//

#include "DQM/SiStripMonitorTrack/interface/SiStripMonitorMuonHLT.h"

using namespace edm;
using namespace reco;
using namespace std;


//
// constructors and destructor
//
SiStripMonitorMuonHLT::SiStripMonitorMuonHLT (const edm::ParameterSet & iConfig)
{
  //now do what ever initialization is needed
  parameters_ = iConfig;
  verbose_ = parameters_.getUntrackedParameter<bool>("verbose",false);
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","HLT/HLTMonMuon");
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt",-1);

  //tags
  clusterCollectionTag_ = parameters_.getUntrackedParameter < InputTag > ("clusterCollectionTag",edm::InputTag("hltSiStripRawToClustersFacility"));
  l3collectionTag_ = parameters_.getUntrackedParameter < InputTag > ("l3MuonTag",edm::InputTag("hltL3MuonCandidates"));
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
  if (outputFile_.size () != 0) LogWarning ("HLTMuonDQMSource") << "Muon HLT Monitoring histograms will be saved to " << outputFile_ << std::endl;
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
  LogDebug ("SiStripMonitorHLTMuon") << " processing conterEvt_: " << counterEvt_ << endl;


  edm::ESHandle < TrackerGeometry > TG;
  iSetup.get < TrackerDigiGeometryRecord > ().get (TG);
  const TrackerGeometry *theTrackerGeometry = TG.product ();
  const TrackerGeometry & theTracker (*theTrackerGeometry);

  Handle < RecoChargedCandidateCollection > l3mucands;
  iEvent.getByLabel (l3collectionTag_, l3mucands);
  RecoChargedCandidateCollection::const_iterator cand;

  Handle < edm::LazyGetter < SiStripCluster > >clusters;
  iEvent.getByLabel (clusterCollectionTag_, clusters);
  edm::LazyGetter < SiStripCluster >::record_iterator clust;

  if (!clusters.failedToGet ())
    {
      for (clust = clusters->begin_record (); clust != clusters->end_record (); ++clust)
	{
	  
	  uint detID = clust->geographicalId ();
	  std::stringstream ss;
	  int layer = tkdetmap_->FindLayer (detID);
	  string label = tkdetmap_->getLayerName (layer);
	  const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
	  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
	  // get the cluster position in local coordinates (cm) 
	  LocalPoint clustlp = topol->localPosition (clust->barycenter ());
	  GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
	  LayerMEMap[label.c_str ()].EtaDistribAllClustersMap->Fill (clustgp.eta ());
	  LayerMEMap[label.c_str ()].PhiDistribAllClustersMap->Fill (clustgp.phi ());
	  LayerMEMap[label.c_str ()].EtaPhiAllClustersMap->Fill (clustgp.eta (), clustgp.phi ());
	}
    }

  if (!l3mucands.failedToGet ())
    {
      for (cand = l3mucands->begin (); cand != l3mucands->end (); ++cand)
	{
	  TrackRef l3tk = cand->get < TrackRef > ();

	  for (size_t hit = 0; hit < l3tk->recHitsSize (); hit++)
	    {
	      //if hit is valid and in tracker say true
	      if (l3tk->recHit (hit)->isValid () == true && l3tk->recHit (hit)->geographicalId ().det () == DetId::Tracker)
		{
		  uint detID = l3tk->recHit (hit)->geographicalId ()();
		  const SiStripRecHit2D *hit2D = dynamic_cast < const SiStripRecHit2D * >(l3tk->recHit (hit).get ());
		  const SiStripMatchedRecHit2D *hitMatched2D = dynamic_cast < const SiStripMatchedRecHit2D * >(l3tk->recHit (hit).get ());
		  const ProjectedSiStripRecHit2D *hitProj2D = dynamic_cast < const ProjectedSiStripRecHit2D * >(l3tk->recHit (hit).get ());


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
		      string label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hit2D->cluster_regional ()->barycenter ());
			      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
			      LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta ());
			      LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi ());
			      LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());
			    }
			}
		    }
		  // if SiStripMatchedRecHit2D  
		  if (hitMatched2D != 0)
		    {
		      //hit mono
		      if (hitMatched2D->monoHit ()->cluster_regional ().isNonnull ())
			{
			  if (hitMatched2D->monoHit ()->cluster_regional ().isAvailable ())
			    {
			      detID = hitMatched2D->monoHit ()->cluster_regional ()->geographicalId ();
			    }
			}
		      int layer = tkdetmap_->FindLayer (detID);
		      string label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hitMatched2D->monoHit ()->cluster_regional ()->barycenter ());
			      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
			      LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta ());
			      LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi ());
			      LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());
			    }
			}

		      //hit stereo
		      if (hitMatched2D->stereoHit ()->cluster_regional ().isNonnull ())
			{
			  if (hitMatched2D->stereoHit ()->cluster_regional ().isAvailable ())
			    {
			      detID = hitMatched2D->stereoHit ()->cluster_regional ()->geographicalId ();
			    }
			}
		      layer = tkdetmap_->FindLayer (detID);
		      label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet2 = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet2 != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet2->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hitMatched2D->stereoHit ()->cluster_regional ()->barycenter ());
			      GlobalPoint clustgp = theGeomDet2->surface ().toGlobal (clustlp);
			      LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta ());
			      LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi ());
			      LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());
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
		      string label = tkdetmap_->getLayerName (layer);
		      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detID));
		      if (theGeomDet != 0)
			{
			  const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
			  if (topol != 0)
			    {
			      // get the cluster position in local coordinates (cm) 
			      LocalPoint clustlp = topol->localPosition (hitProj2D->originalHit ().cluster_regional ()->barycenter ());
			      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
			      LayerMEMap[label.c_str ()].EtaDistribOnTrackClustersMap->Fill (clustgp.eta ());
			      LayerMEMap[label.c_str ()].PhiDistribOnTrackClustersMap->Fill (clustgp.phi ());
			      LayerMEMap[label.c_str ()].EtaPhiOnTrackClustersMap->Fill (clustgp.eta (), clustgp.phi ());
			    }
			}
		    }

		}
	    }			//loop over RecHits
	}			//loop over l3mucands
    }				//if l3seed
}


void
SiStripMonitorMuonHLT::createMEs (const edm::EventSetup & es)
{
  float EtaMax = 2.5;

  //Get the tracker geometry
  edm::ESHandle < TrackerGeometry > TG;
  es.get < TrackerDigiGeometryRecord > ().get (TG);
  const TrackerGeometry *theTrackerGeometry = TG.product ();
  const TrackerGeometry & theTracker (*theTrackerGeometry);
 
 
  //Get list of detectors from tracker
  
  std::vector<DetId> Dets = theTracker.detUnitIds();  
  std::map< string,std::vector<float> > mapTkModulesEta ;
  std::map< string,std::vector<float> > mapTkModulesPhi ;
  
  std::map< string,std::vector<float> > mapTkModulesEtaSorted ;
  std::map< string,std::vector<float> > mapTkModulesPhiSorted ;
  
  int NbModules = 0;

  float StripLength; 
  
  //Loop over modules via Tracker Geometry
  //-----------------------------------------
  for(std::vector<DetId>::iterator detid_iterator =  Dets.begin(); detid_iterator!=Dets.end(); detid_iterator++){
    uint32_t detid = (*detid_iterator)();
    
    if ( (*detid_iterator).null() == true) break;
    if(detid == 0)  break;
    
    // Select the propers detectors - avoid pixels
    const GeomDetUnit * GeomDet = theTracker.idToDetUnit(detid);
    const GeomDet::SubDetector detector = GeomDet->subDetector();
    
    if (detector == GeomDetEnumerators::TIB 
	|| detector == GeomDetEnumerators::TID
	|| detector == GeomDetEnumerators::TEC
	|| detector == GeomDetEnumerators::TOB){
            
      // Get the eta, phi of modules
      int mylayer = tkdetmap_->FindLayer (detid);
      string mylabelHisto = tkdetmap_->getLayerName (mylayer);
      
      const StripGeomDetUnit *theGeomDet = dynamic_cast < const StripGeomDetUnit * >(theTracker.idToDet (detid));
      const StripTopology *topol = dynamic_cast < const StripTopology * >(&(theGeomDet->specificTopology ()));
      
      if (NbModules == 1) StripLength = topol->stripLength();

      // Get the position of the 1st strip in local coordinates (cm) 
      LocalPoint clustlp = topol->localPosition (1.);
      GlobalPoint clustgp = theGeomDet->surface ().toGlobal (clustlp);
      
      float eta = clustgp.eta ();
      float phi = clustgp.phi ();
      
      mapTkModulesEta[mylabelHisto].push_back(eta);
      mapTkModulesPhi[mylabelHisto].push_back(phi);

      mapTkModulesEtaSorted[mylabelHisto].push_back(eta);
      mapTkModulesPhiSorted[mylabelHisto].push_back(phi);

      NbModules++;
    }
    
  }
  
  ////////////////////////////////////////////////////
  ///  Creation of folder structure
  ///    and ME decleration
  ////////////////////////////////////////////////////

  std::string fullName, folder;

  //STRUCTURE OF DETECTORS
  int LayerNumber = 35;
  
  std::vector<float> v_widthPhi(LayerNumber);
  std::vector<bool> v_boolStereo(LayerNumber);
  
  //PHI WIDTH
  //TOB

  std::map< string,float > map_widthPhi ;
  std::map< string,float > map_boolStereo ;
  
  map_widthPhi["TOB_L1"] = (2.*M_PI)/42.;
  map_widthPhi["TOB_L2"] = ((2.*M_PI)/48.);
  map_widthPhi["TOB_L3"] = ((2.*M_PI)/54.);
  map_widthPhi["TOB_L4"] = ((2.*M_PI)/61.);
  map_widthPhi["TOB_L5"] = ((2.*M_PI)/66.);
  map_widthPhi["TOB_L6"] = ((2.*M_PI)/74.);
  
  //TIB
  map_widthPhi["TIB_L1"] = ((2.*M_PI)/30.);
  map_widthPhi["TIB_L2"] = ((2.*M_PI)/34.);
  map_widthPhi["TIB_L3"] = ((2.*M_PI)/48.);
  map_widthPhi["TIB_L4"] = ((2.*M_PI)/53.);
  
  //TIB
  map_widthPhi["TIDM_D1"] = ((2.*M_PI)/24.);
  map_widthPhi["TIDM_D2"] = ((2.*M_PI)/24.);
  map_widthPhi["TIDM_D3"] = ((2.*M_PI)/24.);
  map_widthPhi["TIDP_D1"] = ((2.*M_PI)/24.);
  map_widthPhi["TIDP_D2"] = ((2.*M_PI)/24.);
  map_widthPhi["TIDP_D3"] = ((2.*M_PI)/24.);
  
  //TEC
  map_widthPhi["TECM_W1"] = ((2.*M_PI)/16.);
  map_widthPhi["TECM_W2"] = ((2.*M_PI)/16.);
  map_widthPhi["TECM_W3"] = ((2.*M_PI)/16.);
  map_widthPhi["TECM_W4"] = ((2.*M_PI)/16.);
  map_widthPhi["TECM_W5"] = ((2.*M_PI)/16.);
  map_widthPhi["TECM_W6"] = ((2.*M_PI)/16.);
  map_widthPhi["TECM_W7"] = ((2.*M_PI)/16.);
  map_widthPhi["TECM_W8"] = ((2.*M_PI)/16.);
  map_widthPhi["TECM_W9"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W1"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W2"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W3"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W4"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W5"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W6"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W7"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W8"] = ((2.*M_PI)/16.);
  map_widthPhi["TECP_W9"] = ((2.*M_PI)/16.);
  
  //BOOL STEREO
  //TOB
  map_boolStereo["TOB_L1"] = (true);
  map_boolStereo["TOB_L2"] = (true);
  map_boolStereo["TOB_L3"] = (false);
  map_boolStereo["TOB_L4"] = (false);
  map_boolStereo["TOB_L5"] = (false);
  map_boolStereo["TOB_L6"] = (false);
  
  //TIB
  map_boolStereo["TIB_L1"] = (true);
  map_boolStereo["TIB_L2"] = (true);
  map_boolStereo["TIB_L3"] = (false);
  map_boolStereo["TIB_L4"] = (false);
  
  //TID
  map_boolStereo["TIDM_D1"] = (true);
  map_boolStereo["TIDM_D2"] = (true);
  map_boolStereo["TIDM_D3"] = (true);
  map_boolStereo["TIDP_D1"] = (true);
  map_boolStereo["TIDP_D2"] = (true);
  map_boolStereo["TIDP_D3"] = (true);
  
  //TEC
  map_boolStereo["TECM_W1"] = (false);
  map_boolStereo["TECM_W2"] = (false);
  map_boolStereo["TECM_W3"] = (false);
  map_boolStereo["TECM_W4"] = (false);
  map_boolStereo["TECM_W5"] = (false);
  map_boolStereo["TECM_W6"] = (false);
  map_boolStereo["TECM_W7"] = (false);
  map_boolStereo["TECM_W8"] = (false);
  map_boolStereo["TECM_W9"] = (false);
  map_boolStereo["TECP_W1"] = (false);
  map_boolStereo["TECP_W2"] = (false);
  map_boolStereo["TECP_W3"] = (false);
  map_boolStereo["TECP_W4"] = (false);
  map_boolStereo["TECP_W5"] = (false);
  map_boolStereo["TECP_W6"] = (false);
  map_boolStereo["TECP_W7"] = (false);
  map_boolStereo["TECP_W8"] = (false);
  map_boolStereo["TECP_W9"] = (false);
    
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

      string histoname;
      string title;
      string labelHisto = tkdetmap_->getLayerName (layer);
    
      //Get the proper eta-phi binning

      std::vector<float> vectorPhi;
      std::vector<float> vectorEta;
      std::vector<float> vectorEta_StripPhi;

            
      sort(mapTkModulesEtaSorted[labelHisto].begin(),mapTkModulesEtaSorted[labelHisto].end());
      sort(mapTkModulesPhiSorted[labelHisto].begin(),mapTkModulesPhiSorted[labelHisto].end());

      float PhiFirst = (*(mapTkModulesPhiSorted[labelHisto].begin()));

      float Border1Phi = 0.;
      float Border2Phi = 0.;

      //BARREL : strips in phi
      if (subDet == SiStripDetId::TOB || subDet == SiStripDetId::TIB){
	Border1Phi = PhiFirst - (map_widthPhi[labelHisto]/2.);
	Border2Phi = PhiFirst + (map_widthPhi[labelHisto]/2.);
	
      }
      //ENDCAP : strips in eta
      if (subDet == SiStripDetId::TEC || subDet == SiStripDetId::TID){
	Border1Phi = PhiFirst;
	Border2Phi = PhiFirst + map_widthPhi[labelHisto];
      }
      
      float PhiIter;
      //for barrel if Border1Phi out of range
      if (Border1Phi < -M_PI) {
	PhiIter = Border2Phi;
	Border1Phi = Border2Phi;
	Border2Phi = Border2Phi+map_widthPhi[labelHisto];	
      }
      if (Border1Phi >= -M_PI) PhiIter = Border1Phi;
      
      //BUILD PHI VECTOR
      vectorPhi.push_back(-M_PI);
      while (PhiIter < M_PI){
	vectorPhi.push_back(PhiIter);
	PhiIter = PhiIter + map_widthPhi[labelHisto];
      }
      vectorPhi.push_back(M_PI);
      
      int sizePhi = vectorPhi.size();
      float * xbinsPhi = new float[sizePhi];

      for (int i = 0; i < sizePhi; i++){
	xbinsPhi[i] = vectorPhi[i]; 
      }

      //PUT THE Border1Phi at the border if out of range
      if (Border1Phi < -M_PI) Border1Phi = - M_PI;
     
      //TAG MODULES BETWEEN Border1Phi AND Border2Phi AND SORT THEM
      unsigned int i = 0;
      while (i < mapTkModulesEta[labelHisto].size()){	
	if (Border1Phi < mapTkModulesPhi[labelHisto][i] && mapTkModulesPhi[labelHisto][i] < Border2Phi){
	  vectorEta_StripPhi.push_back(mapTkModulesEta[labelHisto][i]);
	}
	i = i +1;
      }
      sort(vectorEta_StripPhi.begin(),vectorEta_StripPhi.end());

      //MONO OR STEREO
      int step;
      if (map_boolStereo[labelHisto] = false) step = 1;
      if (map_boolStereo[labelHisto] = true) step = 2;

      //BUILD ETA VECTOR
      i = 0;
      vectorEta.push_back(-EtaMax);
      while (i < vectorEta_StripPhi.size()){
	
	//1st bin
	
	float EtaWidth = 0.;
	if ( (i +step) < vectorEta_StripPhi.size()) EtaWidth = vectorEta_StripPhi[i+step] - vectorEta_StripPhi[i];
	if ( (i +step) >= vectorEta_StripPhi.size()) EtaWidth = vectorEta_StripPhi[i] - vectorEta_StripPhi[i-step];
	
	//1st bin

	//BARREL : no shift
	if (subDet == SiStripDetId::TOB || subDet == SiStripDetId::TIB){
	  
	  //1st bin
	  if (i == 0.) vectorEta.push_back(vectorEta_StripPhi[i]-EtaWidth);

	  //FILL IN
	  vectorEta.push_back(vectorEta_StripPhi[i]);

	  //last bin
	  if ( (step == 1 && i == (vectorEta_StripPhi.size()-1))
	      || (step == 2 && (i == (vectorEta_StripPhi.size()-1) || i == (vectorEta_StripPhi.size()-2)) ) 
	       ) vectorEta.push_back(vectorEta_StripPhi[i]+EtaWidth);
	}
	//ENDCAP : shift
	if (subDet == SiStripDetId::TEC || subDet == SiStripDetId::TID){
	  
	  //1st bin
	  if (i == 0.) vectorEta.push_back(vectorEta_StripPhi[i]-(EtaWidth/2.)-EtaWidth);
	  
	  //FILL IN
	  vectorEta.push_back(vectorEta_StripPhi[i]-(EtaWidth/2.) );

	  //last bin
	  if ( (step == 1 && i == (vectorEta_StripPhi.size()-1))
	      || (step == 2 && (i == (vectorEta_StripPhi.size()-1) || i == (vectorEta_StripPhi.size()-2)) ) 
	       ) {
	    vectorEta.push_back(vectorEta_StripPhi[i]+(EtaWidth/2.));
	    vectorEta.push_back(vectorEta_StripPhi[i]+(EtaWidth/2.) + EtaWidth);
	  }
	}
	  
	i = i + step;
      }
      vectorEta.push_back(EtaMax);
      sort(vectorEta.begin(),vectorEta.end()); //for being sure

      int sizeEta = vectorEta.size();
      float * xbinsEta = new float[sizeEta];
      
      for (int i = 0; i < sizeEta; i++){
	xbinsEta[i] = vectorEta[i]; 
      }

      //--------------------------------------------------------------
      //--------------------------------------------------------------
      
      // all clusters
      histoname = "EtaAllClustersDistrib_" + labelHisto;
      title = "#eta(All Clusters) in " + labelHisto;
      layerMEs.EtaDistribAllClustersMap = dbe_->book1D (histoname, title, sizeEta - 1, xbinsEta);
      histoname = "PhiAllClustersDistrib_" + labelHisto;
      title = "#phi(All Clusters) in " + labelHisto;
      layerMEs.PhiDistribAllClustersMap = dbe_->book1D (histoname, title, sizePhi - 1, xbinsPhi);

      histoname = "EtaPhiAllClustersMap_" + labelHisto;
      title = "#eta-#phi All Clusters map in " + labelHisto;
      layerMEs.EtaPhiAllClustersMap = dbe_->book2D (histoname, title, sizeEta - 1, xbinsEta, sizePhi - 1, xbinsPhi);

      // on track clusters
      histoname = "EtaOnTrackClustersDistrib_" + labelHisto;
      title = "#eta(OnTrack Clusters) in " + labelHisto;
      layerMEs.EtaDistribOnTrackClustersMap = dbe_->book1D (histoname, title, sizeEta - 1, xbinsEta);
      histoname = "PhiOnTrackClustersDistrib_" + labelHisto;
      title = "#phi(OnTrack Clusters) in " + labelHisto;
      layerMEs.PhiDistribOnTrackClustersMap = dbe_->book1D (histoname, title, sizePhi - 1, xbinsPhi);
      histoname = "EtaPhiOnTrackClustersMap_" + labelHisto;
      title = "#eta-#phi OnTrack Clusters map in " + labelHisto;
      layerMEs.EtaPhiOnTrackClustersMap = dbe_->book2D (histoname, title, sizeEta - 1, xbinsEta, sizePhi - 1, xbinsPhi);

      LayerMEMap[labelHisto] = layerMEs;
	
    }				//end of loop over layers

  ////////////////////////////////////////////////////


}				//end of method


// ------------ method called once each job just before starting event loop  ------------
void
SiStripMonitorMuonHLT::beginJob (const edm::EventSetup & es)
{
  if (dbe_)
    {
      if (monitorName_ != "")
	monitorName_ = monitorName_ + "/";
      LogInfo ("HLTMuonDQMSource") << "===>DQM event prescale = " << prescaleEvt_ << " events " << endl;
      createMEs (es);
    }
}

// ------------ method called once each job just after ending the event loop  ------------
void
SiStripMonitorMuonHLT::endJob ()
{
  LogInfo ("SiStripMonitorHLTMuon") << "analyzed " << counterEvt_ << " events";
  return;
}

DEFINE_FWK_MODULE(SiStripMonitorMuonHLT);
