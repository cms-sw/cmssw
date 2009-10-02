#include "Alignment/TrackerAlignment/plugins/OverlapTagger.h"

using namespace edm;
using namespace reco;

OverlapTagger::OverlapTagger(const edm::ParameterSet& iConfig):
  src_( iConfig.getParameter<edm::InputTag>("src") ),
  srcClust_( iConfig.getParameter<edm::InputTag>("Clustersrc") ),
  rejectBadMods_(  iConfig.getParameter<bool>("rejectBadMods")),
  BadModsList_(  iConfig.getParameter<std::vector<uint32_t> >("BadMods"))
{

  produces<AliClusterValueMap>(); //produces the ValueMap (VM) to be stored in the Event at the end
}

OverlapTagger::~OverlapTagger(){}


void OverlapTagger::produce(edm::Event &iEvent, const edm::EventSetup &iSetup){
  edm::Handle<TrajTrackAssociationCollection> assoMap;
  iEvent.getByLabel(src_,  assoMap);
  // cout<<"\n\n############################\n###  Starting a new OverlapTagger - Ev "<<iEvent.id().run()<<", "<<iEvent.id().event()<<endl;

  AlignmentClusterFlag iniflag;
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelclusters;
  iEvent.getByLabel(srcClust_,  pixelclusters);//same label as tracks
  std::vector<AlignmentClusterFlag> pixelvalues(pixelclusters->dataSize(), iniflag);//vector where to store value to be fileld in the VM

  edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripclusters;
  iEvent.getByLabel(srcClust_,  stripclusters);//same label as tracks
  std::vector<AlignmentClusterFlag> stripvalues(stripclusters->dataSize(), iniflag);//vector where to store value to be fileld in the VM



  //start doing the thing!

  //loop over trajectories
  for (TrajTrackAssociationCollection::const_iterator itass = assoMap->begin();  itass != assoMap->end(); ++itass){
    
    int nOverlaps=0;
    const edm::Ref<std::vector<Trajectory> >traj = itass->key;//trajectory in the collection
    const Trajectory * myTrajectory= &(*traj);
    std::vector<TrajectoryMeasurement> tmColl =myTrajectory->measurements();

    const reco::TrackRef tkref = itass->val;//associated track track in the collection
    // const Track * trk = &(*tkref);
    int hitcnt=-1;

    //loop over traj meas
    const TrajectoryMeasurement* previousTM(0);
    DetId previousId(0);
    int previousLayer(-1);

    for(std::vector<TrajectoryMeasurement>::const_iterator itTrajMeas = tmColl.begin(); itTrajMeas!=tmColl.end(); ++itTrajMeas){
      hitcnt++;

      if ( previousTM!=0 ) {
	//	std::cout<<"Checking TrajMeas ("<<hitcnt+1<<"):"<<std::endl;
	if(!previousTM->recHit()->isValid()){
	  //std::cout<<"Previous RecHit invalid !"<<std::endl; 
	continue;}
	//else std::cout<<"\nDetId: "<<std::flush<<previousTM->recHit()->geographicalId().rawId()<<"\t Local x of hit: "<<previousTM->recHit()->localPosition().x()<<std::endl;
      }
      else{
	//std::cout<<"This is the first Traj Meas of the Trajectory! The Trajectory contains "<< tmColl.size()<<" TrajMeas"<<std::endl;
      }
      
      
      TransientTrackingRecHit::ConstRecHitPointer hitpointer = itTrajMeas->recHit();
      const TrackingRecHit *hit=&(* hitpointer);
      if(!hit->isValid())continue;

      //std::cout << "         hit number " << (ith - itt->recHitsBegin()) << std::endl;
      DetId detid = hit->geographicalId();
      int layer(layerFromId(detid));//layer 1-4=TIB, layer 5-10=TOB
      int subDet = detid.subdetId();
     

      if ( ( previousTM!=0 )&& (layer!=-1 )) {
	for (std::vector<TrajectoryMeasurement>::const_iterator itmCompare =itTrajMeas-1;itmCompare >= tmColl.begin() &&  itmCompare > itTrajMeas - 4;--itmCompare){
	  DetId compareId = itmCompare->recHit()->geographicalId();
	  if ( subDet != compareId.subdetId() || layer  != layerFromId(compareId)) break;
	  if (!itmCompare->recHit()->isValid()) continue;
	  if ( (subDet<=2) || (subDet > 2 && SiStripDetId(detid).stereo()==SiStripDetId(compareId).stereo())){//if either pixel or strip stereo module
	    
	    //
	    //WOW, we have an overlap!!!!!!
	    //
	    AlignmentClusterFlag hitflag(hit->geographicalId());
	    hitflag.SetOverlapFlag();
	    // cout<<"Overlap found in SubDet "<<subDet<<"!!!"<<flush;
	    
	    if (subDet>2){
	      //cout<<"  TypeId of the RecHit: "<<className(*hit)<<endl;	  
	      const TSiStripRecHit2DLocalPos* transstriphit = dynamic_cast<const  TSiStripRecHit2DLocalPos*>(hit);
	     
	      if(transstriphit!=0){
		const SiStripRecHit2D* striphit=transstriphit->specificHit();   
		//  cout<<"Pointer to SiStripRecHit2D= "<<striphit<<endl;
		SiStripRecHit2D::ClusterRef stripclust(striphit->cluster());
		
		if(stripclust.id()==stripclusters.id()){//ensure that the pixclust is really present in the original cluster collection!!!
		  stripvalues[stripclust.key()]=hitflag;

		  //cout<<">>> Storing in the ValueMap a StripClusterRef with Cluster.Key: "<<stripclust.key()<<" ("<<striphit->cluster().key() <<"), Cluster.Id: "<<stripclust.id()<<"  (DetId is "<<hit->geographicalId().rawId()<<")"<<endl;
		}
		else{
		  cout<<"ERROR in <OverlapTagger::produce>: ProdId of Strip clusters mismatched: "<<stripclust.id()<<" vs "<<stripclusters.id() <<endl;
		}

		// cout<<"Cluster baricentre: "<<stripclust->barycenter()<<endl;
	      }
	      else{
		cout<<"ERROR in <OverlapTagger::produce>: Dynamic cast of Strip RecHit failed!   TypeId of the RecHit: "<<className(*hit)<<endl;
	      }
	    }
	    else {//pixel hit
	      const TSiPixelRecHit* transpixelhit = dynamic_cast<const TSiPixelRecHit*>(hit);
	      if(transpixelhit!=0){
		const SiPixelRecHit* pixelhit=transpixelhit->specificHit();
		SiPixelClusterRefNew pixclust(pixelhit->cluster());
		
		if(pixclust.id()==pixelclusters.id()){
		  pixelvalues[pixclust.key()]=hitflag;
		  //cout<<">>> Storing in the ValueMap a PixelClusterRef with ProdID: "<<pixclust.id()<<"  (DetId is "<<hit->geographicalId().rawId()<<")" <<endl;//"  and  a Val with ID: "<<flag.id()<<endl;
		}
		else{
		  cout<<"ERROR in <OverlapTagger::produce>: ProdId of Pixel clusters mismatched: "<<pixclust.id()<<" vs "<<pixelclusters.id() <<endl;
		}
	      }
	      else{
		cout<<"ERROR in <OverlapTagger::produce>: Dynamic cast of Pixel RecHit failed!   TypeId of the RecHit: "<<className(*hit)<<endl;
	      }
	    }//end 'else' it is a pixel hit

	    nOverlaps++;
	    break;
	  }
	}//end second loop on TM
      }//end if a previous TM exists

      previousTM = &(* itTrajMeas);
      previousId = detid;
      previousLayer = layer;
    }//end loop over traj meas
    //std::cout<<"Found "<<nOverlaps<<" overlaps in this trajectory"<<std::endl; 

  }//end loop over trajectories



  // prepare output 
  std::auto_ptr<AliClusterValueMap> hitvalmap( new AliClusterValueMap);
  AliClusterValueMap::Filler mapfiller(*hitvalmap); 

  edm::TestHandle<std::vector<AlignmentClusterFlag> > fakePixelHandle( &pixelvalues,pixelclusters.id());
  mapfiller.insert(fakePixelHandle, pixelvalues.begin(), pixelvalues.end());

  edm::TestHandle<std::vector<AlignmentClusterFlag> > fakeStripHandle( &stripvalues,stripclusters.id());
  mapfiller.insert(fakeStripHandle, stripvalues.begin(), stripvalues.end());
  mapfiller.fill();





  // iEvent.put(stripmap);
  iEvent.put(hitvalmap);
}//end  OverlapTagger::produce
int OverlapTagger::layerFromId (const DetId& id) const
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

DEFINE_FWK_MODULE(OverlapTagger);
