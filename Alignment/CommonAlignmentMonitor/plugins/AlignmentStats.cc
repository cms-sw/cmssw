#include "Alignment/CommonAlignmentMonitor/plugins/AlignmentStats.h"

//using namespace edm;

AlignmentStats::AlignmentStats(const edm::ParameterSet &iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  overlapAM_(iConfig.getParameter<edm::InputTag>("OverlapAssoMap")),
  keepTrackStats_(iConfig.getParameter<bool>("keepTrackStats")),
  keepHitPopulation_(iConfig.getParameter<bool>("keepHitStats")),
  statstreename_(iConfig.getParameter<string>("TrkStatsFileName")),
  hitstreename_(iConfig.getParameter<string>("HitStatsFileName")),
  prescale_(iConfig.getParameter<uint32_t>("TrkStatsPrescale"))
{

  //sanity checks



}//end constructor

AlignmentStats::~AlignmentStats(){
			       //
}

void AlignmentStats::beginJob( const edm::EventSetup &iSetup){

  //book track stats tree
  treefile_=new TFile(statstreename_.c_str(),"RECREATE");
  treefile_->cd();
  outtree_=new TTree("AlignmentTrackStats","Statistics of Tracks used for Alignment");
  // int nHitsinPXB[MAXTRKS], nHitsinPXE[MAXTRKS], nHitsinTEC[MAXTRKS], nHitsinTIB[MAXTRKS],nHitsinTOB[MAXTRKS],nHitsinTID[MAXTRKS];
  

  outtree_->Branch("Ntracks"    ,&ntracks ,"Ntracks/i");
  outtree_->Branch("Run"        ,&run     ,"RunNr/I");
  outtree_->Branch("Event"      ,&event   ,"EventNr/I");
  outtree_->Branch("Eta"        ,&Eta     ,"Eta[Ntracks]/F");
  outtree_->Branch("Phi"        ,&Phi     ,"Phi[Ntracks]/F");
  outtree_->Branch("P"          ,&P       ,"P[Ntracks]/F");
  outtree_->Branch("Pt"         ,&Pt      ,"Pt[Ntracks]/F");
  outtree_->Branch("Chi2n"      ,&Chi2n   ,"Chi2n[Ntracks]/F");
  outtree_->Branch("Nhits"      ,&Nhits   ,"Nhits[Ntracks][7]/I");
  /*
    outtree_->Branch("NhitsPXB"       , ,);
    outtree_->Branch("NhitsPXE"       , ,);
    outtree_->Branch("NhitsTIB"       , ,);
    outtree_->Branch("NhitsTID"       , ,);
    outtree_->Branch("NhitsTOB"       , ,);
    outtree_->Branch("NhitsTOB"       , ,);
  */

  tmppresc=prescale_;

  //load the tracker geometry from the EventSetup
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry_);
 
}//end beginJob


void AlignmentStats::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup){

  //take trajectories and tracks to loop on
  // edm::Handle<TrajTrackAssociationCollection> TrackAssoMap;
  edm::Handle<reco::TrackCollection> Tracks;
  iEvent.getByLabel(src_, Tracks);

  //take overlap HitAssomap
  edm::Handle<AliClusterValueMap> hMap;
  iEvent.getByLabel(overlapAM_, hMap);
  AliClusterValueMap OverlapMap=*hMap;

  // Initialise
  run=1;
  event=1;
  ntracks=0;
  run=iEvent.id().run();
  event=iEvent.id().event();
  ntracks=Tracks->size();
  //  if(ntracks>1) std::cout<<"~~~~~~~~~~~~\n For this event processing "<<ntracks<<" tracks"<<std::endl;

  unsigned int trk_cnt=0;
  
  for(int j=0;j<MAXTRKS;j++){
    Eta[j]=-9999.0;
    Phi[j]=-8888.0;
    P[j]  =-7777.0;
    Pt[j] =-6666.0;
    Chi2n[j]=-2222.0;
    for(int k=0;k<7;k++){
      Nhits[j][k]=0;
    }
  }
  
  // int npxbhits=0;

  //loop on tracks
  for(std::vector<reco::Track>::const_iterator ittrk = Tracks->begin(), edtrk = Tracks->end(); ittrk != edtrk; ++ittrk){
    Eta[trk_cnt]=ittrk->eta();
    Phi[trk_cnt]=ittrk->phi();
    Chi2n[trk_cnt]=ittrk->normalizedChi2();
    P[trk_cnt]=ittrk->p();
    Pt[trk_cnt]=ittrk->pt();
    Nhits[trk_cnt][0]=ittrk->numberOfValidHits();
 
    //   if(ntracks>1)std::cout<<"Track #"<<trk_cnt+1<<" params:    Eta="<< Eta[trk_cnt]<<"  Phi="<< Phi[trk_cnt]<<"  P="<<P[trk_cnt]<<"   Nhits="<<Nhits[trk_cnt][0]<<std::endl;
   
    int nhit=0;
    //loop on tracking rechits
    //std::cout << "   loop on hits of track #" << (itt - tracks->begin()) << std::endl;
    for (trackingRecHit_iterator ith = ittrk->recHitsBegin(), edh = ittrk->recHitsEnd(); ith != edh; ++ith) {

      const TrackingRecHit *hit = ith->get(); // ith is an iterator on edm::Ref to rechit
      if(! hit->isValid())continue;
      DetId detid = hit->geographicalId();
      int subDet = detid.subdetId();
      uint32_t rawId = hit->geographicalId().rawId();

      //  if(subDet==1)npxbhits++;

      //look if you find this detid in the map
      DetHitMap::iterator mapiter;
      mapiter= hitmap_.find(rawId);
      if(mapiter!=hitmap_.end() ){//present, increase its value by one
	hitmap_[rawId]=hitmap_[rawId]+1;
      }
      else{//not present, let's add this key to the map with value=1
	hitmap_.insert(pair<uint32_t, uint32_t>(rawId, 1));
      }
      
      AlignmentClusterFlag inval;


      //check also if the hit is an overlap. If yes fill a dedicated hitmap
      if (subDet>2){
	//Notice the difference respect to when one loops on Trajectories: the recHit is a TrackingRecHit and not a TransientTrackingRecHit
	const SiStripRecHit2D* striphit=dynamic_cast<const  SiStripRecHit2D*>(hit);
	if(striphit!=0){
	  SiStripRecHit2D::ClusterRef stripclust(striphit->cluster());
	  inval = OverlapMap[stripclust];
	  //cout<<"Taken the Strip Cluster with ProdId "<<stripclust.id() <<"; the Value in the map is "<<inval<<"  (DetId is "<<hit->geographicalId().rawId()<<")"<<endl;
	}
	else{
	  cout<<"ERROR in <AlignmentStats::analyze>: Dynamic cast of Strip RecHit failed!   TypeId of the RecHit: "<<className(*hit)<<endl;
	}

      }//end if hit in Strips
      else{
	const SiPixelRecHit* pixelhit= dynamic_cast<const SiPixelRecHit*>(hit);
	if(pixelhit!=0){
	  SiPixelClusterRefNew pixclust(pixelhit->cluster());
	  inval = OverlapMap[pixclust];
	  //cout<<"Taken the Pixel Cluster with ProdId "<<pixclust.id() <<"; the Value in the map is "<<inval<<"  (DetId is "<<hit->geographicalId().rawId()<<")"<<endl;
	}
	else{
	  cout<<"ERROR in <AlignmentStats::analyze>: Dynamic cast of Pixel RecHit failed!   TypeId of the RecHit: "<<className(*hit)<<endl;
	}
      }//end else hit is in Pixel


      bool isOverlapHit(inval.isOverlap());
      
      if( isOverlapHit ){
	//cout<<"This hit is an overlap !"<<endl;
	DetHitMap::iterator overlapiter;
	overlapiter=overlapmap_.find(rawId);
	
	if(overlapiter!=overlapmap_.end() ){//the det already collected at least an overlap, increase its value by one
	    overlapmap_[rawId]=overlapmap_[rawId]+1;
	}
	else{//first overlap on det unit, let's add it to the map
	  overlapmap_.insert(pair<uint32_t, uint32_t>(rawId, 1));
	}
      }//end if the hit is an overlap
      



      int subdethit= static_cast<int>(hit->geographicalId().subdetId());
      // if(ntracks>1)std::cout<<"Hit in SubDet="<<subdethit<<std::endl;
      Nhits[trk_cnt][subdethit]=Nhits[trk_cnt][subdethit]+1;
      nhit++;
    }//end loop on trackingrechits
    trk_cnt++;

  }//end loop on tracks

  //  cout<<"Total number of pixel hits is "<<npxbhits<<endl;

  tmppresc--;
  if(tmppresc<1){
    outtree_->Fill();
    tmppresc=prescale_;
  }
  if(trk_cnt!=ntracks)std::cout<<"\nERROR! trk_cnt="<<trk_cnt<<"   ntracks="<<ntracks<<std::endl<<std::endl;
  trk_cnt=0;

  return;
}

void AlignmentStats::endJob(){

  treefile_->cd();
  std::cout<<"Writing out the TrackStatistics in "<<gDirectory->GetPath()<<std::endl;
  outtree_->Write();
  delete outtree_;

  //create tree with hit maps (hitstree)
  //book track stats tree
  TFile *hitsfile=new TFile(hitstreename_.c_str(),"RECREATE");
  hitsfile->cd();
  TTree *hitstree=new TTree("AlignmentHitMap","Maps of Hits used for Alignment");

  unsigned int id=0,nhits=0,noverlaps=0;
  float posX(-99999.0),posY(-77777.0),posZ(-88888.0);
  float posEta(-6666.0),posPhi(-5555.0),posR(-4444.0);
  int subdet=0;
  unsigned int layer=0; 
  bool is2D=false,isStereo=false;
  hitstree->Branch("DetId",    &id ,      "DetId/i");
  hitstree->Branch("Nhits",    &nhits ,   "Nhits/i");
  hitstree->Branch("Noverlaps",&noverlaps,"Noverlaps/i");
  hitstree->Branch("SubDet",   &subdet,   "SubDet/I");
  hitstree->Branch("Layer",    &layer,    "Layer/i");
  hitstree->Branch("is2D" ,    &is2D,     "is2D/B");
  hitstree->Branch("isStereo", &isStereo, "isStereo/B");
  hitstree->Branch("posX",     &posX,     "posX/F");
  hitstree->Branch("posY",     &posY,     "posY/F");
  hitstree->Branch("posZ",     &posZ,     "posZ/F");
  hitstree->Branch("posR",     &posR,     "posR/F");
  hitstree->Branch("posEta",   &posEta,   "posEta/F");
  hitstree->Branch("posPhi",   &posPhi,   "posPhi/F");

  /*
  TTree *overlapstree=new TTree("OverlapHitMap","Maps of Overlaps used for Alignment");
  hitstree->Branch("DetId",   &id ,     "DetId/i");
  hitstree->Branch("NOverlaps",   &nhits ,  "Nhits/i");
  hitstree->Branch("SubDet",  &subdet,  "SubDet/I");
  hitstree->Branch("Layer",   &layer,   "Layer/i");
  hitstree->Branch("is2D" ,   &is2D,    "is2D/B");
  hitstree->Branch("isStereo",&isStereo,"isStereo/B");
  hitstree->Branch("posX",    &posX,    "posX/F");
  hitstree->Branch("posY",    &posY,    "posY/F");
  hitstree->Branch("posZ",    &posZ,    "posZ/F");
  hitstree->Branch("posR",    &posR,    "posR/F");
  hitstree->Branch("posEta",  &posEta,  "posEta/F");
  hitstree->Branch("posPhi",  &posPhi,  "posPhi/F");
  */


  AlignableTracker* theAliTracker=new AlignableTracker(&(*trackerGeometry_));
  const std::vector<Alignable*>& Detunitslist=theAliTracker->deepComponents();
  int ndetunits=Detunitslist.size();
  std::cout<<"Number of DetUnits in the AlignableTracker: "<< ndetunits<<std::endl;

  for(int det_cnt=0;det_cnt< ndetunits;++det_cnt){

    //if detunit in vector is found also in the map, look for how many hits were collected 
    //and save in the tree this number
    id=static_cast <uint32_t>( Detunitslist[det_cnt]->id() );
    if( hitmap_.find(id) != hitmap_.end() ){
      nhits=hitmap_[id];
    }
    //if not, save nhits=0
    else{
      nhits=0;
      hitmap_.insert(pair<uint32_t, uint32_t>(id, 0));
    }

    if( overlapmap_.find(id) != overlapmap_.end() ){
      noverlaps=overlapmap_[id];
    }
    //if not, save nhits=0
    else{
      noverlaps=0;
      overlapmap_.insert(pair<uint32_t, uint32_t>(id, 0));
    }

  //take other geometrical infos from the det
   posX= Detunitslist[det_cnt]->globalPosition().x();
   posY= Detunitslist[det_cnt]->globalPosition().y();
   posZ= Detunitslist[det_cnt]->globalPosition().z();

   align::GlobalVector vec(posX,posY,posZ);
   posR = vec.perp();
   posPhi = vec.phi();
   posEta = vec.eta();
   //   posPhi = atan2(posY,posX);	

   DetId detid(id);
   subdet=detid.subdetId();

   //get layers, petals, etc...
   if(subdet==1){//PXB
     PXBDetId pxbdet=(id);
     layer=pxbdet.layer();
     is2D=true;
     isStereo=false;
   }
   else if(subdet==2){
     PXFDetId pxfdet(id);
     layer=pxfdet.disk();
     is2D=true;
     isStereo=false;
   }
   else if(subdet==3){
     TIBDetId tibdet(id);
     layer=tibdet.layerNumber();
     is2D=tibdet.isDoubleSide();
     isStereo=tibdet.isStereo();
   }
   else if(subdet==4){
     TIDDetId tiddet(id);
     layer=tiddet.diskNumber();
     is2D=tiddet.isDoubleSide();
     isStereo=tiddet.isStereo();
   }
   else if(subdet==5){
     TOBDetId tobdet(id);
     layer=tobdet.layerNumber();
     is2D=tobdet.isDoubleSide();
     isStereo=tobdet.isStereo();
   }
   else if(subdet==6){
     TECDetId tecdet(id);
     layer=tecdet.wheelNumber();
     is2D=tecdet.isDoubleSide();
     isStereo=tecdet.isStereo();
   }
   else{
     //exception to be thrown
   }

  //write in the hitstree
    hitstree->Fill();
  }//end loop over detunits


  //save hitstree
  hitstree->Write();
  delete hitstree;
  //delete Detunitslist;
  hitmap_.clear();
  overlapmap_.clear();
  delete hitsfile;
}
// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlignmentStats);
