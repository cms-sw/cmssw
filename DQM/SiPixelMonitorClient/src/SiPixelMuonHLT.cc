// -*- C++ -*-
//
// Package:    SiPixelMuonHLT
// Class:      SiPixelMuonHLT
// 
/**\class 

 Description: Pixel DQM source for Clusters

 Implementation:
     <Notes on implementation>
*/
//////////////////////////////////////////////////////////
//
// Original Author:  Dan Duggan
//         Created:  
// $Id: SiPixelMuonHLT.cc,v 1.9 2012/12/26 21:05:53 wmtan Exp $
//
//////////////////////////////////////////////////////////
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelMuonHLT.h"
#include <string>
#include <stdlib.h>

SiPixelMuonHLT::SiPixelMuonHLT(const edm::ParameterSet& iConfig) :
  conf_(iConfig)
{

  parameters_ = iConfig;

  verbose_ = parameters_.getUntrackedParameter < bool > ("verbose", false);
  monitorName_ = parameters_.getUntrackedParameter < std::string > ("monitorName", "HLT/HLTMonMuon");
  saveOUTput_  = parameters_.getUntrackedParameter < bool > ("saveOUTput", true);

  //tags
  clusterCollectionTag_ = parameters_.getUntrackedParameter < edm::InputTag > ("clusterCollectionTag", edm::InputTag ("hltSiPixelClusters"));
  rechitsCollectionTag_ = parameters_.getUntrackedParameter < edm::InputTag > ("rechitsCollectionTag", edm::InputTag ("hltSiPixelRecHits"));
  l3MuonCollectionTag_  = parameters_.getUntrackedParameter < edm::InputTag > ("l3MuonCollectionTag", edm::InputTag ("hltL3MuonCandidates"));
  //////////////////////////

   theDMBE = edm::Service<DQMStore>().operator->();
   edm::LogInfo ("PixelHLTDQM") << "SiPixelMuonHLT::SiPixelMuonHLT: Got DQM BackEnd interface"<<std::endl;
   outputFile_ = parameters_.getUntrackedParameter < std::string > ("outputFile", "");
   if (outputFile_.size () != 0)
     edm::LogWarning ("HLTMuonDQMSource") << "Muon HLT Monitoring histograms will be saved to " << outputFile_ << std::endl;
   else
     outputFile_ = "PixelHLTDQM.root";
   ///////
   if (theDMBE != NULL) theDMBE->setCurrentFolder (monitorName_); 
   SiPixelMuonHLT::Histo_init();
   
}

SiPixelMuonHLT::~SiPixelMuonHLT()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  edm::LogInfo ("PixelHLTDQM") << "SiPixelMuonHLT::~SiPixelMuonHLT: Destructor"<<std::endl;

}

void SiPixelMuonHLT::beginJob(){

  edm::LogInfo ("PixelHLTDQM") << " SiPixelMuonHLT::beginJob - Initialisation ... " << std::endl;
  eventNo = 0;

}


void SiPixelMuonHLT::endJob(void){
  if(saveOUTput_){
    edm::LogInfo ("PixelHLTDQM") << " SiPixelMuonHLT::endJob - Saving Root File " << std::endl;
    theDMBE->save( outputFile_.c_str() );
  }
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelMuonHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();


  eventNo++;

  edm::ESHandle < TrackerGeometry > TG;
  iSetup.get < TrackerDigiGeometryRecord > ().get (TG);
  const TrackerGeometry *theTrackerGeometry = TG.product ();
  const TrackerGeometry & theTracker (*theTrackerGeometry);
  edm::Handle < reco::RecoChargedCandidateCollection > l3mucands;
  reco::RecoChargedCandidateCollection::const_iterator cand;
  edm::Handle <edmNew::DetSetVector<SiPixelCluster> > clusters;
  edm::Handle <edmNew::DetSetVector<SiPixelRecHit> > rechits;

  bool GotClusters = true;
  bool GotRecHits  = true;
  bool GotL3Muons  = true;
  
  iEvent.getByLabel("hltSiPixelClusters", clusters);
  if(!clusters.isValid()){
    edm::LogInfo("PixelHLTDQM") << "No pix clusters, cannot run for event " << iEvent.eventAuxiliary ().event() <<" run: "<<iEvent.eventAuxiliary ().run()  << std::endl;
    GotClusters = false;
  }
  iEvent.getByLabel("hltSiPixelRecHits", rechits);
  if(!rechits.isValid()){
    edm::LogInfo("PixelHLTDQM") << "No pix rechits, cannot run for event " << iEvent.eventAuxiliary ().event() <<" run: "<<iEvent.eventAuxiliary ().run()  << std::endl;
    GotRecHits = false;
  }
  iEvent.getByLabel (l3MuonCollectionTag_, l3mucands);
  if(!l3mucands.isValid()){
    edm::LogInfo("PixelHLTDQM") << "No L3 Muons, cannot run for event " << iEvent.eventAuxiliary ().event() <<" run: "<<iEvent.eventAuxiliary ().run()  << std::endl;
    GotL3Muons = false;
  }
  
  if (GotClusters){
    if(!clusters.failedToGet ())
      {
	int NBarrel[4] = {0,0,0,0};
	int NEndcap[5] = {0,0,0,0,0};
	for (size_t i = 0; i < clusters->size(); ++i){ 
	  const SiPixelCluster* clust = clusters->data(i);
	  clust->charge();
	  //// Check to see that the detID is correct for each cluster   
	  uint detID = clusters->id(i);
	  const PixelGeomDetUnit *PixGeom = dynamic_cast < const PixelGeomDetUnit * >(theTracker.idToDet (detID));
	  const PixelTopology *topol = dynamic_cast < const PixelTopology * >(&(PixGeom->specificTopology ()));
	  // get the cluster position in local coordinates (cm)
	  LocalPoint clustlp = topol->localPosition (MeasurementPoint(clust->x(),clust->y()));
	  GlobalPoint clustgp = PixGeom->surface ().toGlobal (clustlp);
	  if(PixGeom->geographicalId().subdetId() == 1){ //1 Defines a barrel hit
	    int clustLay = tTopo->pxbLayer(detID);
	    //Eta-Phi
	    MEContainerAllBarrelEtaPhi[0]->Fill(clustgp.eta(),clustgp.phi());
	    MEContainerAllBarrelZPhi[0]->Fill(clustgp.z(),clustgp.phi());
	    MEContainerAllBarrelEtaPhi[clustLay]->Fill(clustgp.eta(),clustgp.phi());
	    MEContainerAllBarrelZPhi[clustLay]->Fill(clustgp.z(),clustgp.phi());
	    //Eta
	    MEContainerAllBarrelEta[0]->Fill(clustgp.eta());
	    MEContainerAllBarrelZ[0]->Fill(clustgp.z());
	    MEContainerAllBarrelEta[clustLay]->Fill(clustgp.eta());
	    MEContainerAllBarrelZ[clustLay]->Fill(clustgp.z());
	    //Phi
	    MEContainerAllBarrelPhi[0]->Fill(clustgp.phi());
	    MEContainerAllBarrelPhi[clustLay]->Fill(clustgp.phi());
	    ++NBarrel[0]; //N clusters all layers
	    ++NBarrel[clustLay]; //N clusters all layers
	  }
	  /////Endcap Pixels ///////
	  if(PixGeom->geographicalId().subdetId() == 2){ //2 Defines a Endcap hit
	    int clustDisk  = tTopo->pxfDisk(detID);
	    if( tTopo->pxfSide(detID) == 2)
	      clustDisk = clustDisk +2;//neg z disks have ID 3 and 4
	    MEContainerAllEndcapXY[0]->Fill(clustgp.x(),clustgp.y());
	    MEContainerAllEndcapXY[clustDisk]->Fill(clustgp.x(),clustgp.y());
	    MEContainerAllEndcapPhi[0]->Fill(clustgp.phi());
	    MEContainerAllEndcapPhi[clustDisk]->Fill(clustgp.phi());
	    ++NEndcap[0];
	    ++NEndcap[clustDisk];
	  }
	  
	}
	MEContainerAllBarrelN[0]->Fill(NBarrel[0]);
	for (int lay = 1; lay < 4; ++lay)
	  MEContainerAllBarrelN[lay]->Fill(NBarrel[lay]);
	MEContainerAllEndcapN[0]->Fill(NEndcap[0]);
	for (int disk = 1; disk < 5; ++disk)
	  MEContainerAllEndcapN[disk]->Fill(NEndcap[disk]); 
      }//if clust (!failedToGet)
  }
  bool doRecHits = false;
  
  if (GotRecHits && doRecHits){
    if(!rechits.failedToGet ())
      {
	for (size_t i = 0; i < rechits->size(); ++i){ 
	  const SiPixelRecHit* myhit = rechits->data(i);
	  uint detID = rechits->id(i);
	  const PixelGeomDetUnit *PixGeom = dynamic_cast < const PixelGeomDetUnit * >(theTracker.idToDet (detID));
	  //edm::LogInfo("PixelHLTDQM") << "" << PixGeom->geographicalId().subdetId() << std::endl;
	  //const PixelTopology *topol = dynamic_cast < const PixelTopology * >(&(PixGeom->specificTopology ()));
	  // get the hit position in local coordinates (cm)
	  //LocalPoint hitlp = topol->localPosition (MeasurementPoint(myhit->x(),myhit->y()));
	  if(PixGeom->geographicalId().subdetId() == 1 && myhit->hasPositionAndError()){
	    GlobalPoint hitgp = PixGeom->surface ().toGlobal (myhit->localPosition());
	    edm::LogInfo("PixelHLTDQM") << " (From SiPixelRecHit) Hit Eta: " << hitgp.eta()   << " Hit Phi: " << hitgp.phi()  << std::endl;
	  }
	}      
      }
  }
  if(GotL3Muons){
    if(!l3mucands.failedToGet ())
      {
	int NBarrel[4] = {0,0,0,0};
	int NEndcap[5] = {0,0,0,0,0};
	for (cand = l3mucands->begin (); cand != l3mucands->end (); ++cand){
	  reco::TrackRef l3tk = cand->get < reco::TrackRef > ();
	  for (size_t hit = 0; hit < l3tk->recHitsSize (); hit++){
	    if (l3tk->recHit (hit)->isValid () == true && l3tk->recHit (hit)->geographicalId ().det () == DetId::Tracker){
	      int detID = l3tk->recHit(hit)->geographicalId().rawId();
	      //if hit is in pixel detector say true
	      bool IdMatch = typeid(*(l3tk->recHit(hit))) == typeid(SiPixelRecHit);
	      if (IdMatch){
		const SiPixelRecHit *pixhit = dynamic_cast < const SiPixelRecHit * >(l3tk->recHit(hit).get());
		if((*pixhit).isValid() == true){
		  edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& pixclust = (*pixhit).cluster();
		  if (!(*pixhit).cluster().isAvailable()) 
		    {continue;}
		  const PixelGeomDetUnit *PixGeom = dynamic_cast < const PixelGeomDetUnit * >(theTracker.idToDet (detID));
		  const PixelTopology *topol = dynamic_cast < const PixelTopology * >(&(PixGeom->specificTopology ()));
		  LocalPoint clustlp = topol->localPosition (MeasurementPoint(pixclust->x(),pixclust->y()));
		  GlobalPoint clustgp = PixGeom->surface ().toGlobal (clustlp);
		  if(l3tk->recHit(hit)->geographicalId().subdetId() == 1){ //1 Defines a barrel hit
		    //get the cluster position in local coordinates (cm)	  
		    int clustLay = tTopo->pxbLayer(detID);
		    MEContainerOnTrackBarrelEtaPhi[0]->Fill(clustgp.eta(),clustgp.phi());
		    MEContainerOnTrackBarrelZPhi[0]->Fill(clustgp.z(),clustgp.phi());
		    MEContainerOnTrackBarrelEtaPhi[clustLay]->Fill(clustgp.eta(),clustgp.phi());
		    MEContainerOnTrackBarrelZPhi[clustLay]->Fill(clustgp.z(),clustgp.phi());
		    MEContainerOnTrackBarrelEta[0]->Fill(clustgp.eta());
		    MEContainerOnTrackBarrelZ[0]->Fill(clustgp.z());
		    MEContainerOnTrackBarrelEta[clustLay]->Fill(clustgp.eta());
		    MEContainerOnTrackBarrelZ[clustLay]->Fill(clustgp.z());
		    MEContainerOnTrackBarrelPhi[0]->Fill(clustgp.phi());
		    MEContainerOnTrackBarrelPhi[clustLay]->Fill(clustgp.phi());
		    ++NBarrel[0];
		    ++NBarrel[clustLay];
		  }//subdet ==1
		  if(l3tk->recHit(hit)->geographicalId().subdetId() == 2){ //2 Defines a Endcap hit
		    int clustDisk  = tTopo->pxfDisk(detID);
		    if( tTopo->pxfDisk(detID) == 2)
		      clustDisk = clustDisk +2;
		    MEContainerOnTrackEndcapXY[0]->Fill(clustgp.x(),clustgp.y());
		    MEContainerOnTrackEndcapXY[clustDisk]->Fill(clustgp.x(),clustgp.y());
		    MEContainerOnTrackEndcapPhi[0]->Fill(clustgp.phi());
		    MEContainerOnTrackEndcapPhi[clustDisk]->Fill(clustgp.phi());
		    ++NEndcap[0];
		    ++NEndcap[clustDisk];
		  }//subdet ==2
		}//pixhit valid
	      }//typeid match
	    }//l3tk->recHit (hit)->isValid () == true
	  }//loop over RecHits
	}//loop over l3mucands
	MEContainerOnTrackBarrelN[0]->Fill(NBarrel[0]);
	for (int lay = 1; lay < 4; ++lay)
	  MEContainerOnTrackBarrelN[lay]->Fill(NBarrel[lay]);
	MEContainerOnTrackEndcapN[0]->Fill(NEndcap[0]);
	for (int disk = 1; disk < 5; ++disk)
	  MEContainerOnTrackEndcapN[disk]->Fill(NEndcap[disk]);
	
      }//if l3mucands  
  }
}

void SiPixelMuonHLT::Histo_init()
{
   monitorName_ = monitorName_+"/SiPixel";
   int NBinsEta = 100;
   int NBinsPhi = 80;
   float EtaMax = 3.0;
   float ZMax   = 28.0;
   int NBinsZ   = 112;
   float PhiMax = 3.142;
   int   NBinsN = 800;
   float NMax   = 800.;
   int   NBinsX = 100;
   int   NBinsY = 100;
   float XMax   = 20.;
   float YMax   = 20.;
   std::string histoname;
   std::string title;

   theDMBE->setCurrentFolder (monitorName_ + "/Barrel");   
   std::string layerLabel[4] = {"All_Layers", "Layer1", "Layer2", "Layer3"};
   for (unsigned int i = 0; i < 4; i++)
     {
       /////////////All Clusters  /////////////////
       ////Barrel//
       //Eta-Phi
       histoname = "EtaPhiAllBarrelMap_" + layerLabel[i];
       title     = "#eta-#phi Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerAllBarrelEtaPhi[i] = theDMBE->book2D (histoname, title, NBinsEta, -EtaMax, EtaMax, NBinsPhi, -PhiMax, PhiMax);
       //Z-Phi
       histoname = "ZPhiAllBarrelMap_" + layerLabel[i];
       title     = "Z-#phi Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerAllBarrelZPhi[i] = theDMBE->book2D (histoname, title, NBinsZ, -ZMax, ZMax, NBinsPhi, -PhiMax, PhiMax);
       //Eta
       histoname = "EtaAllBarrelMap_" + layerLabel[i];
       title     = "#eta Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerAllBarrelEta[i] = theDMBE->book1D (histoname, title, NBinsEta, -EtaMax, EtaMax);
       //Z
       histoname = "ZAllBarrelMap_" + layerLabel[i];
       title     = "Z Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerAllBarrelZ[i] = theDMBE->book1D (histoname, title, NBinsZ, -ZMax, ZMax);
       //Phi
       histoname = "PhiAllBarrelMap_" + layerLabel[i];
       title     = "#phi Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerAllBarrelPhi[i] = theDMBE->book1D (histoname, title, NBinsPhi, -PhiMax, PhiMax);
       //N clusters
       histoname = "NAllBarrelMap_" + layerLabel[i];
       title     = "#phi Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerAllBarrelN[i] = theDMBE->book1D (histoname, title, NBinsN, 0, NMax);
       ////////////On Track Clusters //////////////
       ////Barrel//
       //Eta-Phi
       histoname = "EtaPhiOnTrackBarrelMap_" + layerLabel[i];
       title = "#eta-#phi On Track Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerOnTrackBarrelEtaPhi[i] = theDMBE->book2D (histoname, title, NBinsEta, -EtaMax, EtaMax, NBinsPhi, -PhiMax, PhiMax);
       //Z-Phi
       histoname = "ZPhiOnTrackBarrelMap_" + layerLabel[i];
       title = "Z-#phi On Track Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerOnTrackBarrelZPhi[i] = theDMBE->book2D (histoname, title, NBinsZ, -ZMax, ZMax, NBinsPhi, -PhiMax, PhiMax);
       //Eta
       histoname = "EtaOnTrackBarrelMap_" + layerLabel[i];
       title     = "#eta On Track Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerOnTrackBarrelEta[i] = theDMBE->book1D (histoname, title, NBinsEta, -EtaMax, EtaMax);
       //Z
       histoname = "ZOnTrackBarrelMap_" + layerLabel[i];
       title     = "Z On Track Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerOnTrackBarrelZ[i] = theDMBE->book1D (histoname, title, NBinsZ, -ZMax, ZMax);
       //Phi
       histoname = "PhiOnTrackBarrelMap_" + layerLabel[i];
       title     = "#phi On Track Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerOnTrackBarrelPhi[i] = theDMBE->book1D (histoname, title, NBinsPhi, -PhiMax, PhiMax);
       //N clusters
       histoname = "NOnTrackBarrelMap_" + layerLabel[i];
       title     = "N_{Clusters} On Track Pixel Barrel Cluster Occupancy Map for " + layerLabel[i];
       MEContainerOnTrackBarrelN[i] = theDMBE->book1D (histoname, title, NBinsN, 0, NMax);
     }

   theDMBE->setCurrentFolder (monitorName_ + "/EndCap");
   std::string diskLabel[5] = {"All_Disks", "InnerPosZ", "OuterPosZ", "InnerNegZ", "OuterNegZ"};
   for (int i = 0;i < 5; ++i) 
     {
       /////////////All Clusters  /////////////////
       ////Endcap//
       //XY
       histoname = "XYAllEndcapMap_" + diskLabel[i];
       title     = "X-Y Pixel Endcap Cluster Occupancy Map for " + diskLabel[i];
       MEContainerAllEndcapXY[i] = theDMBE->book2D (histoname, title, NBinsX, -XMax, XMax, NBinsY, -YMax, YMax);
       //Phi
       histoname = "PhiAllEndcapMap_" + diskLabel[i];
       title     = "#phi Pixel Endcap Cluster Occupancy Map for " + diskLabel[i];
       MEContainerAllEndcapPhi[i] = theDMBE->book1D (histoname, title, NBinsPhi, -PhiMax, PhiMax);
       //N clusters
       histoname = "NAllEndcapMap_" + diskLabel[i];
       title     = "#phi Pixel Endcap Cluster Occupancy Map for " + diskLabel[i];
       MEContainerAllEndcapN[i] = theDMBE->book1D (histoname, title, NBinsN, 0, NMax);
       ////////////On Track Clusters //////////////
       ////Endcap//
       //XY
       histoname = "XYOnTrackEndcapMap_" + diskLabel[i];
       title     = "X-Y Pixel Endcap On Track Cluster Occupancy Map for " + diskLabel[i];
       MEContainerOnTrackEndcapXY[i] = theDMBE->book2D (histoname, title, NBinsX, -XMax, XMax, NBinsY, -YMax, YMax);
       //Phi
       histoname = "PhiOnTrackEndcapMap_" + diskLabel[i];
       title     = "#phi Pixel Endcap On Track Cluster Occupancy Map for " + diskLabel[i];
       MEContainerOnTrackEndcapPhi[i] = theDMBE->book1D (histoname, title, NBinsPhi, -PhiMax, PhiMax);
       //N clusters
       histoname = "NOnTrackEndcapMap_" + diskLabel[i];
       title     = "#phi Pixel Endcap On Track Cluster Occupancy Map for " + diskLabel[i];
       MEContainerOnTrackEndcapN[i] = theDMBE->book1D (histoname, title, NBinsN, 0, NMax);
   }
   return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelMuonHLT);
