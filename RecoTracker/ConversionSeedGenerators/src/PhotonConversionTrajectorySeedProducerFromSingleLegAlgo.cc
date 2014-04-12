#include "RecoTracker/ConversionSeedGenerators/interface/PhotonConversionTrajectorySeedProducerFromSingleLegAlgo.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"


//#define debugTSPFSLA

inline double sqr(double a){return a*a;}

PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
PhotonConversionTrajectorySeedProducerFromSingleLegAlgo(const edm::ParameterSet & conf,
	edm::ConsumesCollector && iC)
  :_conf(conf),seedCollection(0),seedCollectionOfSourceTracks(0),
   theHitsGenerator(new CombinedHitPairGeneratorForPhotonConversion(conf.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet"), iC)),
   theRegionProducer(new GlobalTrackingRegionProducerFromBeamSpot(conf.getParameter<edm::ParameterSet>("RegionFactoryPSet"), iC)),
   creatorPSet(conf.getParameter<edm::ParameterSet>("SeedCreatorPSet")),
   theClusterCheck(conf.getParameter<edm::ParameterSet>("ClusterCheckPSet"), iC),
   theSilentOnClusterCheck(conf.getParameter<edm::ParameterSet>("ClusterCheckPSet").getUntrackedParameter<bool>("silentClusterCheck",false)),
   _vtxMinDoF(conf.getParameter<double>("vtxMinDoF")),
   _maxDZSigmas(conf.getParameter<double>("maxDZSigmas")),
   _maxNumSelVtx(conf.getParameter<uint32_t>("maxNumSelVtx")),
   _applyTkVtxConstraint(conf.getParameter<bool>("applyTkVtxConstraint")),
   _countSeedTracks(0),
   _primaryVtxInputTag(conf.getParameter<edm::InputTag>("primaryVerticesTag")),
   _beamSpotInputTag(conf.getParameter<edm::InputTag>("beamSpotInputTag"))
{
  token_vertex      = iC.consumes<reco::VertexCollection>(_primaryVtxInputTag);
  token_bs          = iC.consumes<reco::BeamSpot>(_beamSpotInputTag);
  token_refitter    = iC.consumes<reco::TrackCollection>(_conf.getParameter<edm::InputTag>("TrackRefitter"));
  init();  
}

PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::~PhotonConversionTrajectorySeedProducerFromSingleLegAlgo() {
}

void PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
clear(){
  if(theSeedCreator!=NULL)
    delete theSeedCreator;
}

void PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
init(){
  theSeedCreator    = new SeedForPhotonConversion1Leg(creatorPSet);
}

void PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
analyze(const edm::Event & event, const edm::EventSetup &setup){

  myEsetup = &setup;
  myEvent = &event;

  if(seedCollection!=0)
    delete seedCollection;

  if(seedCollectionOfSourceTracks!=0)
    delete seedCollectionOfSourceTracks;

  seedCollection= new TrajectorySeedCollection();
  seedCollectionOfSourceTracks= new TrajectorySeedCollection();

  size_t clustsOrZero = theClusterCheck.tooManyClusters(event);
  if (clustsOrZero){
    if (!theSilentOnClusterCheck)
      edm::LogError("TooManyClusters") << "Found too many clusters (" << clustsOrZero << "), bailing out.\n";
    return ;
  }


  edm::ESHandle<MagneticField> handleMagField;
  setup.get<IdealMagneticFieldRecord>().get(handleMagField);
  magField = handleMagField.product();
  if (unlikely(magField->inTesla(GlobalPoint(0.,0.,0.)).z()<0.01)) return;

  _IdealHelixParameters.setMagnField(magField);


  event.getByToken(token_vertex, vertexHandle);
  if (!vertexHandle.isValid() || vertexHandle->empty()){
      edm::LogError("PhotonConversionFinderFromTracks") << "Error! Can't get the product primary Vertex Collection "<< _primaryVtxInputTag <<  "\n";
      return;
  }

  event.getByToken(token_bs,recoBeamSpotHandle);
  

  regions = theRegionProducer->regions(event,setup);
  
  //Do the analysis
  loopOnTracks();

 
#ifdef debugTSPFSLA 
  std::stringstream ss;
  ss.str("");
  ss << "\n++++++++++++++++++\n";
  ss << "seed collection size " << seedCollection->size();
  BOOST_FOREACH(TrajectorySeed tjS,*seedCollection){
    po.print(ss, tjS);
  }
  edm::LogInfo("debugTrajSeedFromSingleLeg") << ss.str();
  //-------------------------------------------------
#endif

   // clear memory
  theHitsGenerator->clearLayerCache();
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) delete (*ir);

}


void PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
loopOnTracks(){

  //--- Get Tracks
  myEvent->getByToken(token_refitter, trackCollectionH);

  if(trackCollectionH.isValid()==0){
    edm::LogError("MissingInput")<<" could not find track collecion:"<<_conf.getParameter<edm::InputTag>("TrackRefitter");
    return;
  }
  size_t idx=0, sel=0;
  _countSeedTracks=0;

  ss.str("");
  
  for( reco::TrackCollection::const_iterator tr = trackCollectionH->begin(); 
       tr != trackCollectionH->end(); tr++, idx++) {
    
    // #ifdef debugTSPFSLA 
    //     ss << "\nStuding track Nb " << idx;
    // #endif

    if(rejectTrack(*tr))  continue;
    std::vector<reco::Vertex> selectedPriVtxCompatibleWithTrack;  
    if(!_applyTkVtxConstraint){
      selectedPriVtxCompatibleWithTrack.push_back(*(vertexHandle->begin())); //Same approach as before
    }else{
      if(!selectPriVtxCompatibleWithTrack(*tr,selectedPriVtxCompatibleWithTrack)) continue;
    }

    sel++;
    loopOnPriVtx(*tr,selectedPriVtxCompatibleWithTrack);
  }
#ifdef debugTSPFSLA 
  edm::LogInfo("debugTrajSeedFromSingleLeg") << ss.str();
  edm::LogInfo("debugTrajSeedFromSingleLeg") << "Inspected " << sel << " tracks over " << idx << " tracks. \t # tracks providing at least one seed " << _countSeedTracks ;
#endif
}

bool PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
selectPriVtxCompatibleWithTrack(const reco::Track& tk, std::vector<reco::Vertex>& selectedPriVtxCompatibleWithTrack){
  
  std::vector< std::pair< double, short> > idx;
  short count=-1;

  double cosPhi=tk.px()/tk.pt();
  double sinPhi=tk.py()/tk.pt();
  double sphi2=tk.covariance(2,2);
  double stheta2=tk.covariance(1,1);

  BOOST_FOREACH(const reco::Vertex& vtx,  *vertexHandle){
    count++;
    if(vtx.ndof()<= _vtxMinDoF) continue;

    double _dz= tk.dz(vtx.position());
    double _dzError=tk.dzError();

    double cotTheta=tk.pz()/tk.pt();
    double dx = vtx.position().x();
    double dy = vtx.position().y();
    double sx2=vtx.covariance(0,0);
    double sy2=vtx.covariance(1,1);

    double sxy2= sqr(cosPhi*cotTheta)*sx2+
      sqr(sinPhi*cotTheta)*sy2+
      sqr(cotTheta*(-dx*sinPhi+dy*cosPhi))*sphi2+
      sqr((1+cotTheta*cotTheta)*(dx*cosPhi+dy*sinPhi))*stheta2;
      
    _dzError=sqrt(_dzError*_dzError+vtx.covariance(2,2)+sxy2); //there is a missing component, related to the element (vtx.x*px+vtx.y*py)/pt * pz/pt. since the tk ref point is at the point of closest approach, this scalar product should be almost zero.

#ifdef debugTSPFSLA
    ss << " primary vtx " << vtx.position()  << " \tk vz " << tk.vz() << " vx " << tk.vx() << " vy " << tk.vy() << " pz/pt " << tk.pz()/tk.pt() << " \t dz " << _dz << " \t " << _dzError << " sxy2 "<< sxy2<< " \t dz/dzErr " << _dz/_dzError<< std::endl;
#endif

    if(fabs(_dz)/_dzError > _maxDZSigmas) continue; 
      
    idx.push_back(std::pair<double,short>(fabs(_dz),count));
  }
  if(idx.size()==0) {
#ifdef debugTSPFSLA
    ss << "no vertex selected " << std::endl; 
#endif
    return false;
}
  
  std::stable_sort(idx.begin(),idx.end(),lt_);
  for(size_t i=0;i<_maxNumSelVtx && i<idx.size();++i){
    selectedPriVtxCompatibleWithTrack.push_back((*vertexHandle)[idx[i].second]);
#ifdef debugTSPFSLA
    ss << "selected vtx dz " << idx[0].first << "  position" << idx[0].second << std::endl;
#endif
  }

  return true;
}

void PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
loopOnPriVtx(const reco::Track& tk, const std::vector<reco::Vertex>& selectedPriVtxCompatibleWithTrack){

  bool foundAtLeastASeedCand=false;
  BOOST_FOREACH(const reco::Vertex vtx, selectedPriVtxCompatibleWithTrack){

    math::XYZPoint primaryVertexPoint=math::XYZPoint(vtx.position());
    
    for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
      const TrackingRegion & region = **ir;

#ifdef debugTSPFSLA 
      ss << "[PrintRegion] " << region.print() << std::endl;
#endif
      
      //This if is just for the _countSeedTracks. otherwise 
      //inspectTrack(&tk,region, primaryVertexPoint);
      //would be enough

      if(
	 inspectTrack(&tk,region, primaryVertexPoint)
	 and
	 !foundAtLeastASeedCand
	 ){
	foundAtLeastASeedCand=true;
	_countSeedTracks++; 
      }

    }
  }
}
  
bool PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
rejectTrack(const reco::Track& track){
  
  math::XYZVector beamSpot;
  if(recoBeamSpotHandle.isValid()) {
    beamSpot =  math::XYZVector(recoBeamSpotHandle->position());

    _IdealHelixParameters.setData(&track,beamSpot);   
    if(_IdealHelixParameters.GetTangentPoint().r()==0){
      //this case means a null results on the _IdealHelixParameters side
      return true;
      }
      
    float rMin=2.; //cm
    if(_IdealHelixParameters.GetTangentPoint().rho()<rMin){
      //this case means a track that has the tangent point nearby the primary vertex
      // if the track is primary, this number tends to be the primary vertex itself
      //Rejecting all the potential photon conversions having a "vertex" inside the beampipe
      //We should not miss too much, seen that the conversions at the beam pipe are the better reconstructed
      return true;
    }
  }

  //-------------------------------------------------------
  /*
  float maxPt2=64.; //Cut on pt^2 Indeed doesn't do almost nothing
  if(track.momentum().Perp2() > maxPt2)
    return true;
  */
  //-------------------------------------------------------
  //Cut in the barrel eta region FIXME: to be extended to endcaps
  /*
  float maxEta=1.3; 
  if(fabs(track.eta()) > maxEta)
    return true;
  */
  //-------------------------------------------------------
  //Reject tracks that have a first valid hit in the pixel barrel/endcap layer/disk 1
  //assume that the hits are aligned along momentum
  /*
  const reco::HitPattern& p=track.hitPattern();
  for (int i=0; i<p.numberOfHits(); i++) {
    uint32_t hit = p.getHitPattern(i);
    // if the hit is valid and in pixel barrel, print out the layer
    if (! p.validHitFilter(hit) ) continue;
    if( (p.pixelBarrelHitFilter(hit) || p.pixelEndcapHitFilter(hit))
	&&
	p.getLayer(hit) == 1
	)
      return true;
    else
      break; //because the first valid hit is in a different layer
  }
  */
  //-------------------------------------------------------


  return false;
}


bool PhotonConversionTrajectorySeedProducerFromSingleLegAlgo::
inspectTrack(const reco::Track* track, const TrackingRegion & region, math::XYZPoint& primaryVertexPoint){

  _IdealHelixParameters.setData(track,primaryVertexPoint);   
    
  if (edm::isNotFinite(_IdealHelixParameters.GetTangentPoint().r()) || 
	(_IdealHelixParameters.GetTangentPoint().r()==0)){
    //this case means a null results on the _IdealHelixParameters side
    return false;
  }

  float rMin=3.; //cm
  if(_IdealHelixParameters.GetTangentPoint().rho()<rMin){
    //this case means a track that has the tangent point nearby the primary vertex
    // if the track is primary, this number tends to be the primary vertex itself
    //Rejecting all the potential photon conversions having a "vertex" inside the beampipe
    //We should not miss too much, seen that the conversions at the beam pipe are the better reconstructed
    return false;
  }

  float ptmin = 0.5;
  float originRBound = 3;
  float originZBound  = 3.;

  GlobalPoint originPos;
  originPos = GlobalPoint(_IdealHelixParameters.GetTangentPoint().x(),
			  _IdealHelixParameters.GetTangentPoint().y(),
			  _IdealHelixParameters.GetTangentPoint().z()
			  );
  float cotTheta;
  if( std::abs(_IdealHelixParameters.GetMomentumAtTangentPoint().rho()) > 1.e-4f ){
    cotTheta=_IdealHelixParameters.GetMomentumAtTangentPoint().z()/_IdealHelixParameters.GetMomentumAtTangentPoint().rho();
  }else{
    if(_IdealHelixParameters.GetMomentumAtTangentPoint().z()>0)
      cotTheta=99999.f; 
    else
      cotTheta=-99999.f; 
  }
  GlobalVector originBounds(originRBound,originRBound,originZBound);

  GlobalPoint pvtxPoint(primaryVertexPoint.x(),
			primaryVertexPoint.y(),
			primaryVertexPoint.z()
			);
  ConversionRegion convRegion(originPos, pvtxPoint, cotTheta, track->thetaError(), -1*track->charge());

#ifdef debugTSPFSLA 
  ss << "\nConversion Point " << originPos << " " << originPos.perp() << "\n";
#endif

  const OrderedSeedingHits & hitss = theHitsGenerator->run(convRegion, region, *myEvent, *myEsetup);
  
  unsigned int nHitss =  hitss.size();

  if(nHitss==0)
    return false;

#ifdef debugTSPFSLA 
  ss << "\n nHitss " << nHitss << "\n";
#endif

  if (seedCollection->empty()) seedCollection->reserve(nHitss); // don't do multiple reserves in the case of multiple regions: it would make things even worse
                                                               // as it will cause N re-allocations instead of the normal log(N)/log(2)
  for (unsigned int iHits = 0; iHits < nHitss; ++iHits) { 

#ifdef debugTSPFSLA 
    ss << "\n iHits " << iHits << "\n";
#endif
    const SeedingHitSet & hits =  hitss[iHits];
    //if (!theComparitor || theComparitor->compatible( hits, es) ) {
    //try{
    theSeedCreator->trajectorySeed(*seedCollection,hits, originPos, originBounds, ptmin, *myEsetup,convRegion.cotTheta(),ss);
    //}catch(cms::Exception& er){
    //  edm::LogError("SeedingConversion") << " Problem in the Single Leg Seed creator " <<er.what()<<std::endl;
    //}catch(std::exception& er){
    //  edm::LogError("SeedingConversion") << " Problem in the Single Leg Seed creator " << er.what()<<std::endl;
    //}
  }
  return true;
}


