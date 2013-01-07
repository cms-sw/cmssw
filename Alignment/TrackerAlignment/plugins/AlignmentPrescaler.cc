#include "Alignment/TrackerAlignment/plugins/AlignmentPrescaler.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Utilities/General/interface/ClassName.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"

// #include "Riostream.h"

using namespace std;

AlignmentPrescaler::AlignmentPrescaler(const edm::ParameterSet &iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src")),
  srcQualityMap_(iConfig.getParameter<edm::InputTag>("assomap")),
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

void AlignmentPrescaler::beginJob(){
  //
  std::cout<<"in AlignmentPrescaler::beginJob"<<std::flush;
   fpresc_=new TFile(prescfilename_.c_str(),"READ");
   tpresc_=(TTree*)fpresc_->Get(presctreename_.c_str());
   tpresc_->BuildIndex("DetId");
   tpresc_->SetBranchStatus("*",0);
   tpresc_->SetBranchStatus("DetId",1);
   tpresc_->SetBranchStatus("PrescaleFactor",1);
   tpresc_->SetBranchStatus("PrescaleFactorOverlap",1);
   cout<<" Branches activated "<<std::flush;
   detid_=0;
   hitPrescFactor_=99.0;
   overlapPrescFactor_=88.0;
   
   tpresc_->SetBranchAddress("DetId",&detid_);
   tpresc_->SetBranchAddress("PrescaleFactor",&hitPrescFactor_);
   tpresc_->SetBranchAddress("PrescaleFactorOverlap",&overlapPrescFactor_);
   cout<<" addressed "<<std::flush;   
   myrand_=new TRandom3();
   //   myrand_->SetSeed();
   cout<<" ok "<<std::endl;

}

void AlignmentPrescaler::endJob( ){

  delete tpresc_;
  fpresc_->Close();
  delete fpresc_;
  delete myrand_;
}

void AlignmentPrescaler::produce(edm::Event &iEvent, const edm::EventSetup &iSetup){
  // std::cout<<"\n\n#################\n### Starting the AlignmentPrescaler::produce ; Event: "<<iEvent.id().run() <<", "<<iEvent.id().event()<<std::endl;
  edm::Handle<reco::TrackCollection> Tracks;
  iEvent.getByLabel(src_, Tracks);
 
  //take  HitAssomap
  edm::Handle<AliClusterValueMap> hMap;
  iEvent.getByLabel(srcQualityMap_, hMap);
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
      const SiPixelRecHit*   pixelhit= dynamic_cast<const SiPixelRecHit*>(hit);
      const SiStripRecHit1D* stripHit1D = dynamic_cast<const SiStripRecHit1D*>(hit);
      const SiStripRecHit2D* stripHit2D = dynamic_cast<const SiStripRecHit2D*>(hit);

      AlignmentClusterFlag tmpflag(hit->geographicalId());
      int stripType=0;
      if(subdetId>2){// SST case
	const std::type_info &type = typeid(*hit); 	 
	if (type == typeid(SiStripRecHit1D))	stripType=1;
	else  if (type == typeid(SiStripRecHit2D))	stripType=2;
	else	stripType=3;

	if(stripType==1) { 
	  //	  const SiStripRecHit1D* stripHit1D = dynamic_cast<const SiStripRecHit1D*>(hit);
	  
	  if(stripHit1D!=0){
	    SiStripRecHit1D::ClusterRef stripclust(stripHit1D->cluster());
	    tmpflag=InValMap[stripclust];
	    tmpflag.SetDetId(hit->geographicalId());
	    if(tmpflag.isOverlap())isOverlapHit=true;
	    // std::cout<<"~*~*~* Prescale (1D) for module "<<tmpflag.detId().rawId()<<"("<<InValMap[stripclust].detId().rawId() <<") is "<<hitPrescFactor_<<std::flush;
	    //  if(tmpflag.isOverlap())cout<<" (it is Overlap)"<<endl;
	    // else cout<<endl;
	    
	  }//end if striphit1D!=0
	}
	else if (stripType==2) {
	  //const SiStripRecHit2D* stripHit2D = dynamic_cast<const SiStripRecHit2D*>(hit);
	  if(stripHit2D!=0){
	    SiStripRecHit2D::ClusterRef stripclust(stripHit2D->cluster());
	    tmpflag=InValMap[stripclust];
	    tmpflag.SetDetId(hit->geographicalId());
	    if(tmpflag.isOverlap())isOverlapHit=true;
	    // std::cout<<"~*~*~* Prescale (2D) for module "<<tmpflag.detId().rawId()<<"("<<InValMap[stripclust].detId().rawId() <<") is "<<hitPrescFactor_<<std::flush;
	    //  if(tmpflag.isOverlap())cout<<" (it is Overlap)"<<endl;
	    // else cout<<endl;
	  
	  }//end if striphit2D!=0
	}
      }//end if is a strip hit
      else{
	//	const SiPixelRecHit*   pixelhit= dynamic_cast<const SiPixelRecHit*>(hit);
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
	takeit=(float(myrand_->Rndm())<=overlapPrescFactor_);
      }
      if( !takeit ){
	float rr=float(myrand_->Rndm());
	takeit=(rr<=hitPrescFactor_);
      }
      if(takeit){//HIT TAKEN !
	//cout<<"  DetId="<<tmpdetid<<" taken!"<<flush;
	tmpflag.SetTakenFlag();

	if(subdetId>2){
	  if(stripType==1){
	    SiStripRecHit1D::ClusterRef stripclust(stripHit1D->cluster());
	    InValMap[stripclust]=tmpflag;//.SetTakenFlag();
	  }
	  else if(stripType==2){
	    SiStripRecHit1D::ClusterRef stripclust(stripHit2D->cluster());
	    InValMap[stripclust]=tmpflag;//.SetTakenFlag();
	  }
	  else std::cout<<"Unknown type of strip hit"<<std::endl;
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


int AlignmentPrescaler::layerFromId (const DetId& id, const TrackerTopology* tTopo) const
{
 if ( uint32_t(id.subdetId())==PixelSubdetector::PixelBarrel ) {
    
    return tTopo->pxbLayer(id);
  }
  else if ( uint32_t(id.subdetId())==PixelSubdetector::PixelEndcap ) {
    
    return tTopo->pxfDisk(id) + (3*(tTopo->pxfSide(id)-1));
  }
  else if ( id.subdetId()==StripSubdetector::TIB ) {
    
    return tTopo->tibLayer(id);
  }
  else if ( id.subdetId()==StripSubdetector::TOB ) {
    
    return tTopo->tobLayer(id);
  }
  else if ( id.subdetId()==StripSubdetector::TEC ) {
    
    return tTopo->tecWheel(id) + (9*(tTopo->pxfSide(id)-1));
  }
  else if ( id.subdetId()==StripSubdetector::TID ) {
    
    return tTopo->tidWheel(id) + (3*(tTopo->tidSide(id)-1));
  }
  return -1;

}//end layerfromId

// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlignmentPrescaler);
