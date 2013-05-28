#include "RecoParticleFlow/PFSimProducer/plugins/ConvBremSeedProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

///RECORD NEEDED
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 

///ESHANDLES
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

///COLLECTION
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/ParticleFlowReco/interface/ConvBremSeed.h"
#include "DataFormats/ParticleFlowReco/interface/ConvBremSeedFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

///PROPAGATION TOOLS
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FastSimulation/ParticlePropagator/src/ParticlePropagator.cc"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FastSimulation/TrajectoryManager/interface/InsideBoundsMeasurementEstimator.h"
#include "FastSimulation/TrajectoryManager/interface/LocalMagneticField.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;

ConvBremSeedProducer::ConvBremSeedProducer(const ParameterSet& iConfig):
  conf_(iConfig),
  fieldMap_(0),
  layerMap_(56, static_cast<const DetLayer*>(0)),
  negLayerOffset_(27)
{
  produces<ConvBremSeedCollection>();
}


ConvBremSeedProducer::~ConvBremSeedProducer()
{
 

}


void
ConvBremSeedProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
 
  LogDebug("ConvBremSeedProducerProducer")<<"START event: "<<iEvent.id().event()
					  <<" in run "<<iEvent.id().run();

  float pfmass=  0.0005;

  ///INPUT COLLECTIONS

  ///PF CLUSTERS
  Handle<PFClusterCollection> PfC;
  iEvent.getByLabel(conf_.getParameter<InputTag>("PFClusters") ,PfC);
  const PFClusterCollection& PPP= *(PfC.product());
 
  ///PIXEL
  Handle<SiPixelRecHitCollection> pixelHits;
  iEvent.getByLabel(conf_.getParameter<InputTag>("pixelRecHits") , pixelHits);

  ///STRIP
  Handle<SiStripRecHit2DCollection> rphirecHits;
  iEvent.getByLabel(conf_.getParameter<InputTag>("rphirecHits"),rphirecHits);
  Handle<SiStripRecHit2DCollection> stereorecHits;
  iEvent.getByLabel(conf_.getParameter<InputTag>("stereorecHits"), stereorecHits);
  Handle<SiStripMatchedRecHit2DCollection> matchedrecHits;
  iEvent.getByLabel(conf_.getParameter<InputTag>("matchedrecHits"), matchedrecHits);

  //GSFPFRECTRACKS
  Handle<GsfPFRecTrackCollection> thePfRecTrackCollection;
  iEvent.getByLabel(conf_.getParameter<InputTag>("PFRecTrackLabel"),
		    thePfRecTrackCollection);
  const GsfPFRecTrackCollection PfRTkColl = *(thePfRecTrackCollection.product()); 


  ///OUTPUT COLLECTION
  std::auto_ptr<ConvBremSeedCollection> output(new ConvBremSeedCollection);


  ///INITIALIZE
  vector<pair< TrajectorySeed , pair<GlobalVector,float> > > unclean;
  //TRIPLET OF MODULES TO BE USED FOR SEEDING
  vector< vector< long int > > tripl;
  //LAYER MAP
  initializeLayerMap(); 



  ///LOOP OVER GSF TRACK COLLECTION

  for(unsigned int ipft=0;ipft<PfRTkColl.size();ipft++){
    GsfPFRecTrackRef pft(thePfRecTrackCollection,ipft);
    LogDebug("ConvBremSeedProducerProducer")<<"NEW GsfPFRecTRACK ";
    float eta_br=0;
    unclean.clear();
    tripl.clear();
    vector<int> gc;
    TrackingRecHitRefVector gsfRecHits=pft->gsfTrackRef()->extra()->recHits();
    float pfoutenergy=sqrt((pfmass*pfmass)+pft->gsfTrackRef()->outerMomentum().Mag2());
    XYZTLorentzVector mom =XYZTLorentzVector(pft->gsfTrackRef()->outerMomentum().x(),
					     pft->gsfTrackRef()->outerMomentum().y(),
					     pft->gsfTrackRef()->outerMomentum().z(),
					     pfoutenergy);
    XYZTLorentzVector pos =   XYZTLorentzVector(pft->gsfTrackRef()->outerPosition().x(),
						pft->gsfTrackRef()->outerPosition().y(),
						pft->gsfTrackRef()->outerPosition().z(),
						0.);
    BaseParticlePropagator theOutParticle=BaseParticlePropagator( RawParticle(mom,pos),
								  0,0,B_.z());
    theOutParticle.setCharge(pft->gsfTrackRef()->charge());

    ///FIND THE CLUSTER ASSOCIATED TO THE GSF TRACK
    gc.push_back(GoodCluster(theOutParticle,PPP,0.5));


    vector<PFBrem> brem =(*pft).PFRecBrem();
    vector<PFBrem>::iterator ib=brem.begin();
    vector<PFBrem>::iterator ib_end=brem.end();
    LogDebug("ConvBremSeedProducerProducer")<<"NUMBER OF BREMS "<<brem.size();

    ///LOOP OVER BREM PHOTONS 
    for (;ib!=ib_end;++ib){
    
      XYZTLorentzVector mom=pft->trajectoryPoint(ib->indTrajPoint()).momentum();
      XYZTLorentzVector pos=
	XYZTLorentzVector(
			  pft->trajectoryPoint(ib->indTrajPoint()).position().x(),
			  pft->trajectoryPoint(ib->indTrajPoint()).position().y(),
			  pft->trajectoryPoint(ib->indTrajPoint()).position().z(),
			  0);
      
      ///BREM SELECTION
      if (pos.Rho()>80) continue;
      if ((pos.Rho()>5)&&(fabs(ib->SigmaDeltaP()/ib->DeltaP())>3)) continue;  
      if (fabs(ib->DeltaP())<3) continue;
      eta_br=mom.eta();
      vector< vector< long int > >Idd;
      


      BaseParticlePropagator p( RawParticle(mom,pos),
				0,0,B_.z());
      p.setCharge(0);
      gc.push_back(GoodCluster(p,PPP,0.2));

      ParticlePropagator PP(p,fieldMap_);

      ///LOOP OVER TRACKER LAYER
      list<TrackerLayer>::const_iterator cyliter= geometry_->cylinderBegin();
      for ( ; cyliter != geometry_->cylinderEnd() ; ++cyliter ) {

       ///TRACKER LAYER SELECTION
       if (!(cyliter->sensitive())) continue;
       PP.setPropagationConditions(*cyliter);
       PP.propagate();    
       if (PP.getSuccess()==0) continue;

       ///FIND COMPATIBLE MODULES
       LocalMagneticField mf(PP.getMagneticField());
       AnalyticalPropagator alongProp(&mf, anyDirection);
       InsideBoundsMeasurementEstimator est;
       const DetLayer* tkLayer = detLayer(*cyliter,PP.Z());
       if (&(*tkLayer)==0) continue;
       TrajectoryStateOnSurface trajState = makeTrajectoryState( tkLayer, PP, &mf);
	    
       std::vector<DetWithState> compat 
	 = tkLayer->compatibleDets( trajState, alongProp, est);
       vector <long int> temp;
       if (compat.size()==0) continue;

       for (std::vector<DetWithState>::const_iterator i=compat.begin(); i!=compat.end(); i++) {
	      
	 long int detid=i->first->geographicalId().rawId();
	 
	 if ((tkLayer->subDetector()!=GeomDetEnumerators::PixelBarrel)&&
	     (tkLayer->subDetector()!=GeomDetEnumerators::PixelEndcap)){
	   
	    
	   StDetMatch DetMatch = (rphirecHits.product())->find((detid));
	   MatDetMatch MDetMatch =(matchedrecHits.product())->find((detid));
	   
		
	   long int DetID=(DetMatch != rphirecHits->end())? detid:0;

	   if ((MDetMatch != matchedrecHits->end()) && !MDetMatch->empty()) {
	     long int pii = MDetMatch->begin()->monoId();
	     StDetMatch CDetMatch = (rphirecHits.product())->find((pii));
	     DetID=(CDetMatch != rphirecHits->end())? pii:0;
	     
	   }

	   temp.push_back(DetID);
	   
	 }
	 else{
	   PiDetMatch DetMatch = (pixelHits.product())->find((detid));
	   long int DetID=(DetMatch != pixelHits->end())? detid:0;
	   temp.push_back(DetID);
	
	
	 }
       }

       Idd.push_back(temp);

      }//END TRACKER LAYER LOOP
      if(Idd.size()<2)continue;

      ///MODULE TRIPLETS SELECTION
      for (unsigned int i=0;i<Idd.size()-2;i++){
	for (unsigned int i1=0;i1<Idd[i].size();i1++){
	  for (unsigned int i2=0;i2<Idd[i+1].size();i2++){
	    for (unsigned int i3=0;i3<Idd[i+2].size();i3++){
	      if ((Idd[i][i1]!=0) &&(Idd[i+1][i2]!=0) &&(Idd[i+2][i3]!=0) ){
		vector<long int >tmp;
		tmp.push_back(Idd[i][i1]);  tmp.push_back(Idd[i+1][i2]); tmp.push_back(Idd[i+2][i3]);
		
		bool newTrip=true;
		for (unsigned int iv=0;iv<tripl.size();iv++){
		  if((tripl[iv][0]==tmp[0])&&(tripl[iv][1]==tmp[1])&&(tripl[iv][2]==tmp[2])) newTrip=false;

		}
		if (newTrip){

		  tripl.push_back(tmp);
		}
	      }
	    }	  
	  }	  
	}
      }
    }//END BREM LOOP

    float sineta_brem =sinh(eta_br);
    

    //OUTPUT COLLECTION
    edm::ESHandle<MagneticField> bfield;
    iSetup.get<IdealMagneticFieldRecord>().get(bfield);
    float nomField = bfield->nominalValue();
 

    TransientTrackingRecHit::ConstRecHitContainer glob_hits;
    OwnVector<TrackingRecHit> loc_hits;
    for (unsigned int i=0;i<tripl.size();i++){
      StDetMatch DetMatch1 = (rphirecHits.product())->find(tripl[i][0]);
      StDetMatch DetMatch2 = (rphirecHits.product())->find(tripl[i][1]);
      StDetMatch DetMatch3 = (rphirecHits.product())->find(tripl[i][2]);
      if ((DetMatch1 == rphirecHits->end()) ||
          (DetMatch2 == rphirecHits->end()) ||
          (DetMatch3 == rphirecHits->end()) )  continue;
      StDetSet DetSet1 = *DetMatch1;
      StDetSet DetSet2 = *DetMatch2;
      StDetSet DetSet3 = *DetMatch3;

      for (StDetSet::const_iterator it1=DetSet1.begin();it1!=DetSet1.end();++it1){
	GlobalPoint gp1=tracker_->idToDet(tripl[i][0])->surface().
	  toGlobal(it1->localPosition());

	bool tak1=isGsfTrack(gsfRecHits,&(*it1));
	
	for (StDetSet::const_iterator it2=DetSet2.begin();it2!=DetSet2.end();++it2){
	  GlobalPoint gp2=tracker_->idToDet(tripl[i][1])->surface().
	    toGlobal(it2->localPosition());
	  bool tak2=isGsfTrack(gsfRecHits,&(*it2));

	  for (StDetSet::const_iterator it3=DetSet3.begin();it3!=DetSet3.end();++it3){
	    //  ips++;
	    GlobalPoint gp3=tracker_->idToDet(tripl[i][2])->surface().
	      toGlobal(it3->localPosition());	 
	    bool tak3=isGsfTrack(gsfRecHits,&(*it3));  
	    

	    FastHelix helix(gp3, gp2, gp1,nomField,&*bfield);
	    GlobalVector gv=helix.stateAtVertex().momentum();
	    GlobalVector gv_corr(gv.x(),gv.y(),gv.perp()*sineta_brem);
	    float ene= sqrt(gv_corr.mag2()+(pfmass*pfmass));

	    GlobalPoint gp=helix.stateAtVertex().position();
	    float ch=helix.stateAtVertex().charge();




	    XYZTLorentzVector mom = XYZTLorentzVector(gv.x(),gv.y(),gv_corr.z(),ene);
	    XYZTLorentzVector pos = XYZTLorentzVector(gp.x(),gp.y(),gp.z(),0.);
	    BaseParticlePropagator theOutParticle(RawParticle(mom,pos),0,0,B_.z());
	    theOutParticle.setCharge(ch);
	    int bgc=GoodCluster(theOutParticle,PPP,0.3,true);

	    if (gv.perp()<0.5) continue;

	    if (tak1+tak2+tak3>2) continue;

	    if (bgc==-1) continue;
	    bool clTak=false;
	    for (unsigned int igcc=0; igcc<gc.size(); igcc++){
	      if (clTak) continue;
	      if (bgc==gc[igcc]) clTak=true;
	    }
	    if (clTak) continue;




	    GlobalTrajectoryParameters Gtp(gp1,
					   gv,int(ch), 
					   &(*magfield_));
	    glob_hits.clear(); loc_hits.clear();
	    glob_hits.push_back(hitBuilder_->build(it1->clone()));
	    glob_hits.push_back(hitBuilder_->build(it2->clone()));
	    glob_hits.push_back(hitBuilder_->build(it3->clone()));

	    ///SEED CREATION

	    FreeTrajectoryState CSeed(Gtp,
				      CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));
	    TrajectoryStateOnSurface updatedState;	  
  
            for (int ih=0;ih<3;ih++){

	      TrajectoryStateOnSurface state = (ih==0)?
		propagator_->propagate(CSeed,
				       tracker_->idToDet(tripl[i][ih])->surface()):
		propagator_->propagate(updatedState,
				       tracker_->idToDet(tripl[i][ih])->surface());
	      
	      if (!state.isValid()){ 
		ih=3;
		continue;}

	      updatedState =  kfUpdator_->update(state, *glob_hits[ih]);
	      loc_hits.push_back(glob_hits[ih]->hit()->clone());
	      if (ih==2){
		PTrajectoryStateOnDet const & PTraj= 
		  trajectoryStateTransform::persistentState(updatedState,tripl[i][2]);
		//		output->push_back(Trajectoryseed(PTraj,loc_hits,alongMomentum));
		unclean.push_back(make_pair(TrajectorySeed(PTraj,loc_hits,alongMomentum), 
 					    make_pair(gv_corr,ch)));
	      }
	      //    }

	    }
	  }    
	}    
      }
    }
    vector<bool> inPhot = sharedHits(unclean);
    for (unsigned int iu=0; iu<unclean.size();iu++){

      if (inPhot[iu])
	output->push_back(ConvBremSeed(unclean[iu].first,pft));

    }

  } //END GSF TRACK COLLECTION LOOP 
  LogDebug("ConvBremSeedProducerProducer")<<"END";
  iEvent.put(output);
    
}


void 
ConvBremSeedProducer::beginRun(const edm::Run& run,
			       const EventSetup& iSetup)
{
  ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
  geomSearchTracker_ = track.product();
  
  ESHandle<TrackerInteractionGeometry>  theTrackerInteractionGeometry;
  iSetup.get<TrackerInteractionGeometryRecord>().get(theTrackerInteractionGeometry );
  geometry_=theTrackerInteractionGeometry.product();

  ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  tracker_=tracker.product();

  ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);
  magfield_=magfield.product();
  B_=magfield_->inTesla(GlobalPoint(0,0,0));

  ESHandle<MagneticFieldMap> fieldMap;
  iSetup.get<MagneticFieldMapRecord>().get(fieldMap);
  fieldMap_ =fieldMap.product();



  ESHandle<TransientTrackingRecHitBuilder> hitBuilder;
  iSetup.get<TransientRecHitRecord>().get(conf_.getParameter<string>("TTRHBuilder"),hitBuilder);
  hitBuilder_= hitBuilder.product();

  propagator_  =    new PropagatorWithMaterial(alongMomentum,0.0005,&(*magfield) );
  kfUpdator_   =    new KFUpdator();
}

void 
ConvBremSeedProducer::endRun(const edm::Run& run,
			     const EventSetup& iSetup) {
  delete propagator_;
  delete kfUpdator_;
}

void 
ConvBremSeedProducer::initializeLayerMap()
{


  // These are the BoundSurface&, the BoundDisk* and the BoundCylinder* for that layer
  //   const BoundSurface& theSurface = layer.surface();
  //   BoundDisk* theDisk = layer.disk();  // non zero for endcaps
  //   BoundCylinder* theCylinder = layer.cylinder(); // non zero for barrel
  //   int theLayer = layer.layerNumber(); // 1->3 PixB, 4->5 PixD, 
  //                                       // 6->9 TIB, 10->12 TID, 
  //                                       // 13->18 TOB, 19->27 TEC
  
  /// ATTENTION: HARD CODED LOGIC! If Famos layer numbering changes this logic needs to 
  /// be adapted to the new numbering!
    
    std::vector< BarrelDetLayer*>   barrelLayers = 
      geomSearchTracker_->barrelLayers();
    LogDebug("FastTracker") << "Barrel DetLayer dump: ";
    for (std::vector< BarrelDetLayer*>::const_iterator bl=barrelLayers.begin();
	 bl != barrelLayers.end(); ++bl) {
      LogDebug("FastTracker")<< "radius " << (**bl).specificSurface().radius(); 
    }

  std::vector< ForwardDetLayer*>  posForwardLayers = 
    geomSearchTracker_->posForwardLayers();
  LogDebug("FastTracker") << "Positive Forward DetLayer dump: ";
  for (std::vector< ForwardDetLayer*>::const_iterator fl=posForwardLayers.begin();
       fl != posForwardLayers.end(); ++fl) {
    LogDebug("FastTracker") << "Z pos "
			    << (**fl).surface().position().z()
			    << " radii " 
			    << (**fl).specificSurface().innerRadius() 
			    << ", " 
			    << (**fl).specificSurface().outerRadius(); 
  }

  const float rTolerance = 1.5;
  const float zTolerance = 3.;

  LogDebug("FastTracker")<< "Dump of TrackerInteractionGeometry cylinders:";
  for( std::list<TrackerLayer>::const_iterator i=geometry_->cylinderBegin();
       i!=geometry_->cylinderEnd(); ++i) {
    const BoundCylinder* cyl = i->cylinder();
    const BoundDisk* disk = i->disk();

    LogDebug("FastTracker") << "Famos Layer no " << i->layerNumber()
			    << " is sensitive? " << i->sensitive()
			    << " pos " << i->surface().position();
    if (!i->sensitive()) continue;

    if (cyl != 0) {

      LogDebug("FastTracker") << " cylinder radius " << cyl->radius();
      bool found = false;

      for (std::vector< BarrelDetLayer*>::const_iterator 
	     bl=barrelLayers.begin(); bl != barrelLayers.end(); ++bl) {

	if (fabs( cyl->radius() - (**bl).specificSurface().radius()) < rTolerance) {

	  layerMap_[i->layerNumber()] = *bl;
	  found = true;
	  LogDebug("FastTracker")<< "Corresponding DetLayer found with radius "
				 << (**bl).specificSurface().radius();
		  
	  break;
	}
      }
      if (!found) {
	LogError("FastTracker") << "FAILED to find a corresponding DetLayer!";
      }
    }
    else {
      LogDebug("FastTracker") << " disk radii " << disk->innerRadius() 
		 << ", " << disk->outerRadius();

      bool found = false;

      for (std::vector< ForwardDetLayer*>::const_iterator fl=posForwardLayers.begin();
	   fl != posForwardLayers.end(); ++fl) {
	if (fabs( disk->position().z() - (**fl).surface().position().z()) < zTolerance) {
	  layerMap_[i->layerNumber()] = *fl;
	  found = true;
	  LogDebug("FastTracker") << "Corresponding DetLayer found with Z pos "
				  << (**fl).surface().position().z()
				  << " and radii " 
				  << (**fl).specificSurface().innerRadius() 
				  << ", " 
				  << (**fl).specificSurface().outerRadius(); 
	  break;
	}
      }
      if (!found) {
	LogError("FastTracker") << "FAILED to find a corresponding DetLayer!";
      }
    }
  }

}
const DetLayer*  ConvBremSeedProducer::detLayer( const TrackerLayer& layer, float zpos) const
{
  if (zpos > 0 || !layer.forward() ) return layerMap_[layer.layerNumber()];
  else return layerMap_[layer.layerNumber()+negLayerOffset_];
}

TrajectoryStateOnSurface 
ConvBremSeedProducer::makeTrajectoryState( const DetLayer* layer, 
			       const ParticlePropagator& pp,
			       const MagneticField* field) const
{

  GlobalPoint  pos( pp.X(), pp.Y(), pp.Z());
  GlobalVector mom( pp.Px(), pp.Py(), pp.Pz());

  ReferenceCountingPointer<TangentPlane> plane = layer->surface().tangentPlane(pos);

  return TrajectoryStateOnSurface
    (GlobalTrajectoryParameters( pos, mom, TrackCharge( pp.charge()), field), *plane);
}
bool ConvBremSeedProducer::isGsfTrack(const TrackingRecHitRefVector& tkv, const TrackingRecHit *h ){
  trackingRecHit_iterator ib=tkv.begin();
  trackingRecHit_iterator ie=tkv.end();
  bool istaken=false;
  //  for (;ib!=ie-2;++ib){
    for (;ib!=ie;++ib){
    if (istaken) continue;
    if (!((*ib)->isValid())) continue;
 
    istaken = (*ib)->sharesInput(h,TrackingRecHit::all);
  }
  return istaken;
}
vector<bool> ConvBremSeedProducer::sharedHits( const vector<pair< TrajectorySeed, 
				   pair<GlobalVector,float> > >& unclean){

  vector<bool> goodseed;
  goodseed.clear();
  if (unclean.size()<2){
    for (unsigned int i=0;i<unclean.size();i++)
      goodseed.push_back(true);
  }else{
 
    for (unsigned int i=0;i<unclean.size();i++)
      goodseed.push_back(true);
    
    for (unsigned int iu=0; iu<unclean.size()-1;iu++){   
      if (!goodseed[iu]) continue;
      for (unsigned int iu2=iu+1; iu2<unclean.size();iu2++){
	if (!goodseed[iu]) continue;
	if (!goodseed[iu2]) continue;
      //    if (unclean[iu].second.second *unclean[iu2].second.second >0)continue;
	
	TrajectorySeed::const_iterator sh = unclean[iu].first.recHits().first;
	TrajectorySeed::const_iterator sh_end = unclean[iu].first.recHits().second; 
	
	unsigned int shar =0;
	for (;sh!=sh_end;++sh){ 

	  TrajectorySeed::const_iterator sh2 = unclean[iu2].first.recHits().first;
	  TrajectorySeed::const_iterator sh2_end = unclean[iu2].first.recHits().second; 
	for (;sh2!=sh2_end;++sh2){

	  if ((*sh).sharesInput(&(*sh2),TrackingRecHit::all))

	    shar++;

	}
	}
	if (shar>=2){
	if (unclean[iu].second.first.perp()<unclean[iu2].second.first.perp()) goodseed[iu]=false;
	else goodseed[iu2]=false;
	}
      
      } 
 
    }
  }
  return goodseed;
}



int ConvBremSeedProducer::GoodCluster(const BaseParticlePropagator& ubpg, const PFClusterCollection& pfc, float minep, bool sec){
  
  BaseParticlePropagator bpg = ubpg;
  bpg.propagateToEcalEntrance(false);
  float dr=1000;
  float de=1000;
  float df=1000;
  int ibest=-1;

  if(bpg.getSuccess()!=0){

    for (unsigned int i =0; i<pfc.size();i++ ){
      float tmp_ep=pfc[i].energy()/bpg.momentum().e();
      float tmp_phi=fabs(pfc[i].position().phi()-bpg.vertex().phi());
      if (tmp_phi>TMath::TwoPi()) tmp_phi-= TMath::TwoPi(); 
      float tmp_eta=fabs(pfc[i].position().eta()-bpg.vertex().eta());
      float tmp_dr=sqrt(pow(tmp_phi,2)+pow(tmp_eta,2));
      bool isBet=(tmp_dr<dr);
      if (sec) isBet=(tmp_phi<df);
      if ((isBet)&&(tmp_ep>minep)&&(tmp_ep<10)){
	dr=tmp_dr;
	de=tmp_eta;
	df=tmp_phi; 
	ibest=i;
      }
    }
    bool isBad=(dr>0.1);
    if (sec) isBad= ((df>0.25) || (de>0.5));

    if (isBad) ibest=-1;

  }
  return ibest;
}

