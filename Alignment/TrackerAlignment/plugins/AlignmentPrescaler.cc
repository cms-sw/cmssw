#include "Alignment/TrackerAlignment/plugins/AlignmentPrescaler.h"


AlignmentPrescaler::AlignmentPrescaler(const edm::ParameterSet &iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src")),
  AM_(iConfig.getParameter<edm::InputTag>("assomap")),
  prescfilename_(iConfig.getParameter<std::string>("PrescFileName")),
  presctreename_(iConfig.getParameter<std::string>("PrescTreeName"))
{
  // issue the produce<>
  produces<AliClusterValueMap>();
  produces<AliTrackTakenClusterValueMap>();

}

AlignmentPrescaler::~AlignmentPrescaler(){
  //  
}

void AlignmentPrescaler::beginJob( const edm::EventSetup & ){
  //
   fpresc_=new TFile(prescfilename_.c_str(),"READ");
   tpresc_=(TTree*)fpresc_->Get(presctreename_.c_str());
   tpresc_->BuildIndex("DetId");
   tpresc_->SetBranchStatus("*",0);
   tpresc_->SetBranchStatus("DetId",1);
   tpresc_->SetBranchStatus("PrescaleFactor",1);
   tpresc_->SetBranchStatus("PrescaleFactorOverlap",1);

   detid_=0;
   preschit_=99.0;
   prescoverlap_=88.0;
   
   tpresc_->SetBranchAddress("DetId",&detid_);
   tpresc_->SetBranchAddress("PrescaleFactor",&preschit_);
   tpresc_->SetBranchAddress("PrescaleFactorOverlap",&prescoverlap_);
   
   myrand=new TRandom3();
   //   myrand->SetSeed();


   /*
   fh=new TFile("debugfile1.root","RECREATE");
   hrr=new TH1F("randdistr","distribution of random numbers",100,0.0,1.0);
   hnhitssubdet=new TH1F("hitvssubdet","Distribution of hits Vs SubDet",7,0.0,7.0);
   hnhitsTIBL3=new TH1I("hitvsphiTIBL3","Distribution of hits Vs DetId (TIB L3 only)",5848,369153044,369158892);
   
   totnhitspxl_=0;
   */


}

void AlignmentPrescaler::endJob( ){
  /*
  cout<<"\n\n%%%%%%%% At the end of AlignmentPrescale the number of PIXEL HITS TAKEN was "<< totnhitspxl_<<endl<<endl;

  fh->cd();
  hrr->Write();
  hnhitssubdet->Write();
  hnhitsTIBL3->Write();


  delete hrr;
  fh->Close();
  delete fh;
  */

  //
  delete tpresc_;
  fpresc_->Close();
  delete fpresc_;
  delete myrand;
}

void AlignmentPrescaler::produce(edm::Event &iEvent, const edm::EventSetup &iSetup){
  //  std::cout<<"\n\n#################\n### Starting the AlignmentPrescaler::produce ; Event: "<<iEvent.id().run() <<", "<<iEvent.id().event()<<std::endl;
  edm::Handle<reco::TrackCollection> Tracks;
  iEvent.getByLabel(src_, Tracks);
 
  //take  HitAssomap
  edm::Handle<AliClusterValueMap> hMap;
  iEvent.getByLabel(AM_, hMap);
  AliClusterValueMap InValMap=*hMap;

  //prepare the output of the ValueMap flagging tracks
  std::vector<int> trackflags(Tracks->size(),0);


  //int npxlhits=0;
  
    //loop on tracks
  for(std::vector<reco::Track>::const_iterator ittrk = Tracks->begin(), edtrk = Tracks->end(); ittrk != edtrk; ++ittrk){
    //loop on tracking rechits
    // std::cout << "Loop on hits of track #" << (ittrk - Tracks->begin()) << std::endl;
    int nhit=0;
    int ntakenhits=0;
    bool firstTakenHit=false;

    for (trackingRecHit_iterator ith = ittrk->recHitsBegin(), edh = ittrk->recHitsEnd(); ith != edh; ++ith) {
      const TrackingRecHit *hit = ith->get(); // ith is an iterator on edm::Ref to rechit
      if(! hit->isValid()){
       	nhit++;
	continue;
      }
      uint32_t tmpdetid = hit->geographicalId().rawId();
      tpresc_->GetEntryWithIndex(tmpdetid);
      

      //-------------
      //decide whether to take this hit or not
      bool takeit=false;  
      int subdetId=hit->geographicalId().subdetId();   
 

      //check first if the cluster is also in the overlap asso map
      bool isOverlapHit=false;
      //  bool first=true;
      //ugly...
      const SiStripRecHit2D* striphit = dynamic_cast<const SiStripRecHit2D*>(hit);
      const SiPixelRecHit*   pixelhit= dynamic_cast<const SiPixelRecHit*>(hit);
      AlignmentClusterFlag tmpflag(hit->geographicalId());
      if(subdetId>2){// SST case
	if(striphit!=0){
	  SiStripRecHit2D::ClusterRef stripclust(striphit->cluster());
	  tmpflag=InValMap[stripclust];
	  tmpflag.SetDetId(hit->geographicalId());
	  if(tmpflag.isOverlap())isOverlapHit=true;
	  // cout<<"~*~*~* Prescale for module "<<tmpflag.detId().rawId()<<"("<<InValMap[stripclust].detId().rawId() <<") is "<<preschit_<<flush;
	  //if(tmpflag.isOverlap())cout<<" (it is Overlap)"<<flush;
	  //	  else cout<<endl;
	  
	}//end if striphit!=0
      }//end if is a strip hit
      else{
	if(pixelhit!=0){
	  //npxlhits++;
	  SiPixelClusterRefNew pixclust(pixelhit->cluster());
	  tmpflag=InValMap[pixclust];
	  tmpflag.SetDetId(hit->geographicalId());
	  if(tmpflag.isOverlap())isOverlapHit=true;
	}
      }//end else is a pixel hit
      //      tmpflag.SetDetId(hit->geographicalId());

      if( isOverlapHit ){
	//cout<<"  DetId="<<tmpdetid<<" is Overlap! "<<flush;
	takeit=(float(myrand->Rndm())<=prescoverlap_);
      }
      if( !takeit ){
	float rr=float(myrand->Rndm());
	takeit=(rr<=preschit_);
      }
      if(takeit){//HIT TAKEN !
	//cout<<"  DetId="<<tmpdetid<<" taken!"<<flush;
	tmpflag.SetTakenFlag();

	if(subdetId>2){
	  SiStripRecHit2D::ClusterRef stripclust(striphit->cluster());
	  InValMap[stripclust]=tmpflag;//.SetTakenFlag();
	  
	}
	else{
	  SiPixelClusterRefNew pixclust(pixelhit->cluster());
	  InValMap[pixclust]=tmpflag;//.SetTakenFlag();
	}
	
	if(!firstTakenHit){
	  firstTakenHit=true;
	  //std::cout<<"Index of the track iterator is "<< ittrk-Tracks->begin() <<endl;
	  
	}
	ntakenhits++;
      }//end if take this hit
      //cout<<endl;

        nhit++;
      //cout<<endl;
    }//end loop on RecHits
    trackflags[ittrk-Tracks->begin()]=ntakenhits;
    // cout<<"Entrioes in debug histo: "<<hrr->GetEntries()<<endl;
  }//end loop on tracks
  


  // totnhitspxl_+=ntakenhits;
  //cout<<"AlignmentPrescaler::produce says that in this event "<<ntakenhits<<" pixel clusters were taken (out of "<<npxlhits<<" total pixel hits."<<endl;



  //save the asso map, tracks...
  // prepare output 
  std::auto_ptr<AliClusterValueMap> OutVM( new AliClusterValueMap);
  *OutVM=InValMap;

  iEvent.put(OutVM);
  
  
  std::auto_ptr<AliTrackTakenClusterValueMap> trkVM( new AliTrackTakenClusterValueMap);
  AliTrackTakenClusterValueMap::Filler trkmapfiller(*trkVM);
  trkmapfiller.insert(Tracks,trackflags.begin(),trackflags.end() );
  trkmapfiller.fill();
  iEvent.put(trkVM);


}//end produce


int AlignmentPrescaler::layerFromId (const DetId& id) const
{
 if ( uint32_t(id.subdetId())==PixelSubdetector::PixelBarrel ) {
    PXBDetId tobId(id);
    return tobId.layer();
  }
  else if ( uint32_t(id.subdetId())==PixelSubdetector::PixelEndcap ) {
    PXFDetId tobId(id);
    return tobId.disk() + (3*(tobId.side()-1));
  }
  else if ( id.subdetId()==StripSubdetector::TIB ) {
    TIBDetId tibId(id);
    return tibId.layer();
  }
  else if ( id.subdetId()==StripSubdetector::TOB ) {
    TOBDetId tobId(id);
    return tobId.layer();
  }
  else if ( id.subdetId()==StripSubdetector::TEC ) {
    TECDetId tobId(id);
    return tobId.wheel() + (9*(tobId.side()-1));
  }
  else if ( id.subdetId()==StripSubdetector::TID ) {
    TIDDetId tobId(id);
    return tobId.wheel() + (3*(tobId.side()-1));
  }
  return -1;

}//end layerfromId

// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlignmentPrescaler);
