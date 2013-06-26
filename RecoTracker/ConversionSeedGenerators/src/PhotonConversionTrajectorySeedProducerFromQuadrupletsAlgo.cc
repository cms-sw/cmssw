#include "RecoTracker/ConversionSeedGenerators/interface/PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo.h"
#include "RecoTracker/ConversionSeedGenerators/interface/Quad.h"
#include "FWCore/Utilities/interface/Exception.h"
// ClusterShapeIncludes:::
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
/*
To Do:

assign the parameters to some data member to avoid search at every event

 */

//#define debugTSPFSLA
//#define mydebug_knuenz

PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo::
PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo(const edm::ParameterSet & conf)
  :_conf(conf),seedCollection(0),
   hitsfactoryPSet(conf.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet")),   
   creatorPSet(conf.getParameter<edm::ParameterSet>("SeedCreatorPSet")),
   regfactoryPSet(conf.getParameter<edm::ParameterSet>("RegionFactoryPSet")),
   theClusterCheck(conf.getParameter<edm::ParameterSet>("ClusterCheckPSet")),
   SeedComparitorPSet(conf.getParameter<edm::ParameterSet>("SeedComparitorPSet")),
   QuadCutPSet(conf.getParameter<edm::ParameterSet>("QuadCutPSet")),
   theSilentOnClusterCheck(conf.getParameter<edm::ParameterSet>("ClusterCheckPSet").getUntrackedParameter<bool>("silentClusterCheck",false)){

  init();  
}
     
void PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo::
clear(){
  if(theHitsGenerator!=NULL)
    delete theHitsGenerator;
  if(theSeedCreator!=NULL)
    delete theSeedCreator;
  if(theRegionProducer!=NULL)
    delete theRegionProducer;
}

void PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo::
init(){
  theHitsGenerator  = new CombinedHitQuadrupletGeneratorForPhotonConversion(hitsfactoryPSet);
  theSeedCreator    = new SeedForPhotonConversionFromQuadruplets(creatorPSet);
  theRegionProducer = new GlobalTrackingRegionProducerFromBeamSpot(regfactoryPSet);
}

void PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo::
analyze(const edm::Event & event, const edm::EventSetup &setup){

  myEsetup = &setup;
  myEvent = &event;

  if(seedCollection!=0)
    delete seedCollection;

  seedCollection= new TrajectorySeedCollection();

  size_t clustsOrZero = theClusterCheck.tooManyClusters(event);
  if (clustsOrZero){
    if (!theSilentOnClusterCheck)
      edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.\n";
    return ;
  }

  regions = theRegionProducer->regions(event,setup);

  event.getByLabel(_conf.getParameter<edm::InputTag>("primaryVerticesTag"), vertexHandle);
  if (!vertexHandle.isValid()){ 
    edm::LogError("PhotonConversionFinderFromTracks") << "Error! Can't get the product primary Vertex Collection "<< _conf.getParameter<edm::InputTag>("primaryVerticesTag") <<  "\n";
    return;
  }

  //Do the analysis
  loop();

#ifdef mydebug_knuenz
  std::cout << "Running PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo" <<std::endl;
#endif

#ifdef debugTSPFSLA 
  std::stringstream ss;
  ss.str("");
  ss << "\n++++++++++++++++++\n";
  ss << "seed collection size " << seedCollection->size();
  BOOST_FOREACH(TrajectorySeed tjS,*seedCollection){
    po.print(ss, tjS);
  }
  edm::LogInfo("debugTrajSeedFromQuadruplets") << ss.str();
  //-------------------------------------------------
#endif
}


void PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo::
loop(){

  
  ss.str("");
  
  float ptMin=0.1;
  
  BOOST_FOREACH(const reco::Vertex primaryVertex, *vertexHandle){

    //FIXME currnetly using the first primaryVertex, should loop on the promaryVertexes
    GlobalTrackingRegion region(ptMin
				,GlobalPoint(
					     primaryVertex.position().x(),
					     primaryVertex.position().y(), 
					     primaryVertex.position().z() 
					     )
				,primaryVertex.xError()*10
				,primaryVertex.zError()*10
				,true
				); 
    
#ifdef debugTSPFSLA 
    ss << "[PrintRegion] " << region.print() << std::endl;
#endif
    
    inspect(region);

  }
#ifdef debugTSPFSLA 
  edm::LogInfo("debugTrajSeedFromQuadruplets") << ss.str();
#endif
}
  

bool PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo::
inspect(const TrackingRegion & region ){

  const OrderedSeedingHits & hitss = theHitsGenerator->run(region, *myEvent, *myEsetup);
  
  unsigned int nHitss =  hitss.size();

#ifdef debugTSPFSLA 
  ss << "\n nHitss " << nHitss << "\n";
#endif

  if (seedCollection->empty()) seedCollection->reserve(nHitss/2); // don't do multiple reserves in the case of multiple regions: it would make things even worse
                                                               // as it will cause N re-allocations instead of the normal log(N)/log(2)

  unsigned int iHits=0, jHits=1;

  //
  // Trivial arbitration
  //
  // Vector to store old quad values
  std::vector<Quad> quadVector;


  for (; iHits < nHitss && jHits < nHitss; iHits+=2 , jHits+=2) { 

#ifdef debugTSPFSLA 
    //    ss << "\n iHits " << iHits << " " << jHits << "\n";
#endif
    //phits is the pair of hits for the positron
    const SeedingHitSet & phits =  hitss[iHits];
    //mhits is the pair of hits for the electron
    const SeedingHitSet & mhits =  hitss[jHits];


    try{
      //FIXME (modify the interface of the seed generator if needed)
      //passing the region, that is centered around the primary vertex
      theSeedCreator->trajectorySeed(*seedCollection, phits, mhits, region, *myEsetup, ss, quadVector, SeedComparitorPSet, QuadCutPSet);
    }catch(cms::Exception& er){
      edm::LogError("SeedingConversion") << " Problem in the Quad Seed creator " <<er.what()<<std::endl;
    }catch(std::exception& er){
      edm::LogError("SeedingConversion") << " Problem in the Quad Seed creator " <<er.what()<<std::endl;
    }
  }
  quadVector.clear();
  return true;
}
