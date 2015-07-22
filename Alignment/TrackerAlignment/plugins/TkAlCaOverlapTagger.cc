#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "Utilities/General/interface/ClassName.h"

#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
//#include <boost/regex.hpp>

class TkAlCaOverlapTagger : public edm::EDProducer {
 public:
  TkAlCaOverlapTagger(const edm::ParameterSet &iConfig);
  ~TkAlCaOverlapTagger();
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

 private:
  edm::InputTag src_;
  edm::InputTag srcClust_;
  bool rejectBadMods_;
  std::vector<unsigned int> BadModsList_;


  int layerFromId (const DetId& id, const TrackerTopology* tTopo) const;
};

TkAlCaOverlapTagger::TkAlCaOverlapTagger(const edm::ParameterSet& iConfig):
  src_( iConfig.getParameter<edm::InputTag>("src") ),
  srcClust_( iConfig.getParameter<edm::InputTag>("Clustersrc") ),
  rejectBadMods_(  iConfig.getParameter<bool>("rejectBadMods")),
  BadModsList_(  iConfig.getParameter<std::vector<uint32_t> >("BadMods"))
{

  produces<AliClusterValueMap>(); //produces the ValueMap (VM) to be stored in the Event at the end
}

TkAlCaOverlapTagger::~TkAlCaOverlapTagger(){}


void TkAlCaOverlapTagger::produce(edm::Event &iEvent, const edm::EventSetup &iSetup){
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::Handle<TrajTrackAssociationCollection> assoMap;
  iEvent.getByLabel(src_,  assoMap);
  // cout<<"\n\n############################\n###  Starting a new TkAlCaOverlapTagger - Ev "<<iEvent.id().run()<<", "<<iEvent.id().event()<<endl;

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
      
      
      TrackingRecHit::ConstRecHitPointer hitpointer = itTrajMeas->recHit();
      const TrackingRecHit *hit=&(* hitpointer);
      if(!hit->isValid())continue;

      //std::cout << "         hit number " << (ith - itt->recHitsBegin()) << std::endl;
      DetId detid = hit->geographicalId();
      int layer(layerFromId(detid, tTopo));//layer 1-4=TIB, layer 5-10=TOB
      int subDet = detid.subdetId();

      if ( ( previousTM!=0 )&& (layer!=-1 )) {
	for (std::vector<TrajectoryMeasurement>::const_iterator itmCompare =itTrajMeas-1;itmCompare >= tmColl.begin() &&  itmCompare > itTrajMeas - 4;--itmCompare){
	  DetId compareId = itmCompare->recHit()->geographicalId();
	  if ( subDet != compareId.subdetId() || layer  != layerFromId(compareId, tTopo)) break;
	  if (!itmCompare->recHit()->isValid()) continue;
	  if ( (subDet<=2) || (subDet > 2 && SiStripDetId(detid).stereo()==SiStripDetId(compareId).stereo())){//if either pixel or strip stereo module
	    
	    //
	    //WOW, we have an overlap!!!!!!
	    //
	    AlignmentClusterFlag hitflag(hit->geographicalId());
	    hitflag.SetOverlapFlag();
	    // cout<<"Overlap found in SubDet "<<subDet<<"!!!"<<flush;

	    bool hitInStrip=(subDet==SiStripDetId::TIB) || (subDet==SiStripDetId::TID) ||(subDet==SiStripDetId::TOB) ||(subDet==SiStripDetId::TEC);
	    if (hitInStrip){
	      //cout<<"  TypeId of the RecHit: "<<className(*hit)<<endl;
	      // const std::type_info &type = typeid(*hit);
	      const SiStripRecHit2D* transstriphit2D = dynamic_cast<const  SiStripRecHit2D*>(hit);
	      const SiStripRecHit1D* transstriphit1D = dynamic_cast<const  SiStripRecHit1D*>(hit);
	   
	      //   if (type == typeid(SiStripRecHit1D)) {
	      if(transstriphit1D!=0){
		//	const SiStripRecHit1D* striphit=dynamic_cast<const  SiStripRecHit1D*>(hit);
		const SiStripRecHit1D* striphit=transstriphit1D;
		if(striphit!=0){
		  SiStripRecHit1D::ClusterRef stripclust(striphit->cluster());
		  
		  if(stripclust.id()==stripclusters.id()){//ensure that the stripclust is really present in the original cluster collection!!!
		    stripvalues[stripclust.key()]=hitflag;
		  }
		  else{
		    edm::LogError("TkAlCaOverlapTagger")<<"ERROR in <TkAlCaOverlapTagger::produce>: ProdId of Strip clusters mismatched: "<<stripclust.id()<<" vs "<<stripclusters.id();
		  }
		}
		else{
		  edm::LogError("TkAlCaOverlapTagger")<<"ERROR in <TkAlCaOverlapTagger::produce>: Dynamic cast of Strip RecHit failed!   TypeId of the RecHit: "<<className(*hit);
		}
	      }//end if sistriprechit1D
	      else if(transstriphit2D!=0){
	      //else if (type == typeid(SiStripRecHit2D)) {
		//		const SiStripRecHit2D* striphit=dynamic_cast<const  SiStripRecHit2D*>(hit);
		const SiStripRecHit2D* striphit=transstriphit2D;   
		if(striphit!=0){
		  SiStripRecHit2D::ClusterRef stripclust(striphit->cluster());
		  
		  if(stripclust.id()==stripclusters.id()){//ensure that the stripclust is really present in the original cluster collection!!!
		    stripvalues[stripclust.key()]=hitflag;
	      
		    //cout<<">>> Storing in the ValueMap a StripClusterRef with Cluster.Key: "<<stripclust.key()<<" ("<<striphit->cluster().key() <<"), Cluster.Id: "<<stripclust.id()<<"  (DetId is "<<hit->geographicalId().rawId()<<")"<<endl;
		  }
		  else{
		    edm::LogError("TkAlCaOverlapTagger")<<"ERROR in <TkAlCaOverlapTagger::produce>: ProdId of Strip clusters mismatched: "<<stripclust.id()<<" vs "<<stripclusters.id();
		  }
		  
		  // cout<<"Cluster baricentre: "<<stripclust->barycenter()<<endl;
		}
		else{
		  edm::LogError("TkAlCaOverlapTagger")<<"ERROR in <TkAlCaOverlapTagger::produce>: Dynamic cast of Strip RecHit failed!   TypeId of the RecHit: "<<className(*hit);
		}
	      }//end if Sistriprechit2D
	      else{
		edm::LogError("TkAlCaOverlapTagger")<<"ERROR in <TkAlCaOverlapTagger::produce>: Impossible to determine the type of SiStripRecHit.  TypeId of the RecHit: "<<className(*hit);
	      }	  
	 
	    }//end if hit in Strips
	    else {//pixel hit
	      const SiPixelRecHit* transpixelhit = dynamic_cast<const SiPixelRecHit*>(hit);
	      if(transpixelhit!=0){
		const SiPixelRecHit* pixelhit=transpixelhit;
		SiPixelClusterRefNew pixclust(pixelhit->cluster());
		
		if(pixclust.id()==pixelclusters.id()){
		  pixelvalues[pixclust.key()]=hitflag;
		  //cout<<">>> Storing in the ValueMap a PixelClusterRef with ProdID: "<<pixclust.id()<<"  (DetId is "<<hit->geographicalId().rawId()<<")" <<endl;//"  and  a Val with ID: "<<flag.id()<<endl;
		}
		else{
		  edm::LogError("TkAlCaOverlapTagger")<<"ERROR in <TkAlCaOverlapTagger::produce>: ProdId of Pixel clusters mismatched: "<<pixclust.id()<<" vs "<<pixelclusters.id();
		}
	      }
	      else{
		edm::LogError("TkAlCaOverlapTagger")<<"ERROR in <TkAlCaOverlapTagger::produce>: Dynamic cast of Pixel RecHit failed!   TypeId of the RecHit: "<<className(*hit);
	      }
	    }//end 'else' it is a pixel hit

	    nOverlaps++;
	    break;
	  }
	}//end second loop on TM
      }//end if a previous TM exists

      previousTM = &(* itTrajMeas);
      previousId = detid;
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
}//end  TkAlCaOverlapTagger::produce
int TkAlCaOverlapTagger::layerFromId (const DetId& id, const TrackerTopology* tTopo) const
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
    
    return tTopo->tecWheel(id) + (9*(tTopo->tecSide(id)-1));
  }
  else if ( id.subdetId()==StripSubdetector::TID ) {
    
    return tTopo->tidWheel(id) + (3*(tTopo->tidSide(id)-1));
  }
  return -1;

}//end layerfromId

// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TkAlCaOverlapTagger);
