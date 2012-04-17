#include "CommonTools/RecoUtils/interface/PF_PU_AssoMapAlgos.h"

#include <vector>
#include <string>

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"

#include "TMath.h"
   
using namespace edm;
using namespace std;
using namespace reco;

const double eMass = 0.000511;
const double kMass = 0.497;
const double piMass = 0.1396;

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;

  typedef pair<TrackRef, float> TrackQualityPair;
  typedef vector< TrackQualityPair > TrackQualityPairVector;
  typedef pair<VertexRef, TrackQualityPair> VertexTrackQuality;

  typedef pair <VertexRef, float>  VertexPtsumPair;
  typedef vector< VertexPtsumPair > VertexPtsumVector;

  typedef math::XYZTLorentzVector LorentzVector;

/*************************************************************************************/
/* function to find the vertex with the highest TrackWeight for a certain track      */ 
/*************************************************************************************/

VertexTrackQuality 
PF_PU_AssoMapAlgos::TrackWeightAssociation(const TrackRef&  trackRef, Handle<VertexCollection> vtxcollH) 
{

	VertexRef bestvertexref(vtxcollH,0);		
 	float bestweight = 0.;

	const TrackBaseRef& trackbaseRef = TrackBaseRef(trackRef);

	//loop over all vertices in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);

     	  //get the most probable vertex for the track
	  float weight = vertexref->trackWeight(trackbaseRef);
	  if(weight>bestweight){
  	    bestweight = weight;
	    bestvertexref = vertexref;
 	  } 

	}

  	return make_pair(bestvertexref,make_pair(trackRef,bestweight));

}


/*************************************************************************************/
/* function to find the closest vertex in z for a certain point                      */ 
/*************************************************************************************/

VertexRef 
PF_PU_AssoMapAlgos::FindClosestInZ(double ztrack, Handle<VertexCollection> vtxcollH)
{

	VertexRef bestvertexref;

	double dzmin = 5.;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);
 
	  //find and store the closest vertex in z
          double dz = fabs(ztrack - vertexref->z());
          if(dz<dzmin) {
            dzmin = dz; 
            bestvertexref = vertexref;
          }
	
	}

	if(dzmin<5.) return bestvertexref;
	  else return VertexRef(vtxcollH,0);
}


/*************************************************************************************/
/* function to associate the track to the closest vertex in z                        */ 
/*************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::AssociateClosestInZ(TrackRef trackref, Handle<VertexCollection> vtxcollH)
{
	return make_pair(PF_PU_AssoMapAlgos::FindClosestInZ(trackref->referencePoint().z(),vtxcollH),make_pair(trackref,-1.));
}


/*******************************************************************************************/
/* function to associate the track to the closest vertex in 3D, absolue distance or sigma  */ 
/*******************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::AssociateClosest3D(TrackRef trackref, Handle<VertexCollection> vtxcollH, 
				       const edm::EventSetup& iSetup, bool input_VertexAssUseAbsDistance_)
{

	VertexRef bestvertexref(vtxcollH,0);

	ESHandle<MagneticField> bFieldHandle;
	iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);

	TransientTrack genTrkTT(trackref, &(*bFieldHandle) );

  	double IpMin = 10.;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);
	        
	  double genTrk3DIp = 10001.;
	  double genTrk3DIpSig = 10001.;
	  pair<bool,Measurement1D> genTrk3DIpPair = IPTools::absoluteImpactParameter3D(genTrkTT, *vertexref);

	  if(genTrk3DIpPair.first){
    	    genTrk3DIp = genTrk3DIpPair.second.value();
	    genTrk3DIpSig = genTrk3DIpPair.second.significance();
	  }
 
	  //find and store the closest vertex
	  if(input_VertexAssUseAbsDistance_){
            if(genTrk3DIp<IpMin){
              IpMin = genTrk3DIp; 
              bestvertexref = vertexref;
            }
	  }else{
            if(genTrk3DIpSig<IpMin){
              IpMin = genTrk3DIpSig; 
              bestvertexref = vertexref;
            }
	  }

        }	

	return make_pair(bestvertexref,make_pair(trackref,-1.));
}


/*************************************************************************************/
/* function to find out if the track comes from a gamma conversion                   */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::ComesFromConversion(const TrackRef trackref, Handle<ConversionCollection> convCollH, Conversion* gamma)
{

 	if(trackref->trackerExpectedHitsInner().numberOfLostHits()>=0){

	  for(unsigned int convcoll_ite=0; convcoll_ite<convCollH->size(); convcoll_ite++){
	
	    if(ConversionTools::matchesConversion(trackref,convCollH->at(convcoll_ite))){
	
	      *gamma = convCollH->at(convcoll_ite);
	      return true;

	    }

	  }

	}

	return false;
}

bool
PF_PU_AssoMapAlgos::FindRelatedElectron(const TrackRef trackref, Handle<GsfElectronCollection> gsfcollH, Handle<TrackCollection> CTFtrkcoll) 
{

	bool output = false;

	TrackRef electrkref;

	//loop over all tracks in the gsfelectron collection
  	for(GsfElectronCollection::const_iterator gsf_ite=gsfcollH->begin(); gsf_ite!=gsfcollH->end(); ++gsf_ite){

	  electrkref = gsf_ite->closestCtfTrackRef();

	  //if gsfelectron's closestCtfTrack is a null reference 
   	  if(electrkref.isNull()){

     	    unsigned int index_trck=0;
     	    int ibest=-1;
     	    unsigned int sharedhits_max=0;
     	    float dr_min=1000;
     	    
	    //search the general track that shares the most hits with the electron seed
     	    for(TrackCollection::const_iterator trck_ite=CTFtrkcoll->begin(); trck_ite!=CTFtrkcoll->end(); ++trck_ite,++index_trck){
       
	      unsigned int sharedhits=0;
       
              float dph= fabs(trck_ite->phi()-gsf_ite->phi()); 
       	      if(dph>TMath::Pi()) dph-= TMath::TwoPi();
       	      float det=fabs(trck_ite->eta()-gsf_ite->eta());
       	      float dr =sqrt(dph*dph+det*det);  
              
	      //loop over all valid hits of the chosen general track
       	      for(trackingRecHit_iterator trackhit_ite=trck_ite->recHitsBegin();trackhit_ite!=trck_ite->recHitsEnd();++trackhit_ite){

	        if(!(*trackhit_ite)->isValid()) continue;

	         //loop over all valid hits of the electron seed 
             	for(TrajectorySeed::const_iterator gsfhit_ite= gsf_ite->gsfTrack()->extra()->seedRef()->recHits().first;gsfhit_ite!=gsf_ite->gsfTrack()->extra()->seedRef()->recHits().second;++gsfhit_ite){
           	  
		  if(!(gsfhit_ite->isValid())) continue;
           	  if((*trackhit_ite)->sharesInput(&*(gsfhit_ite),TrackingRecHit::all))  sharedhits++; 
         	
		}       
         
       	      }
       
 
       	      if((sharedhits>sharedhits_max) || ((sharedhits==sharedhits_max)&&(dr<dr_min))){

                sharedhits_max=sharedhits;
                dr_min=dr;
                ibest=index_trck;

       	      }

    	    }

      	    electrkref = TrackRef(CTFtrkcoll,ibest);

   	  }//END OF if gsfelectron's closestCtfTrack is a null reference

	  if((electrkref->theta()==trackref->theta()) && 
	     (electrkref->phi()==trackref->phi()) && 
	     (electrkref->p()==trackref->p())) output = true;

	}

	return output;
}


/*************************************************************************************/
/* function to find the closest vertex in z for a track from a conversion            */ 
/*************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::FindConversionVertex(const TrackRef trackref, Conversion gamma, Handle<VertexCollection> vtxcollH,
					  const edm::EventSetup& iSetup, bool ass_type)
{  

	VertexRef bestvertexref(vtxcollH,0);

	ESHandle<TransientTrackBuilder> theTTB;
  	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTB); 

	ESHandle<MagneticField> bFieldHandle;
	iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle); 

	GlobalVector phoMomentum;
	if(gamma.nTracks()==2){ 
	  phoMomentum = GlobalVector( gamma.pairMomentum().x(),
				      gamma.pairMomentum().y(),
				      gamma.pairMomentum().z() );
	}else{ 
	  phoMomentum = GlobalVector( trackref->innerMomentum().x(),
				      trackref->innerMomentum().y(),
				      trackref->innerMomentum().z() );
	}

	const math::XYZPoint conversionPoint = gamma.conversionVertex().position();
	const math::XYZVector photonVector( phoMomentum.x(),phoMomentum.y(),phoMomentum.z() );

	Track photon = Track(trackref->chi2(),trackref->ndof(),conversionPoint,photonVector,0,trackref->innerStateCovariance());
	TransientTrack genPhoTT(photon, &(*bFieldHandle) );
    
        KinematicParticleFactoryFromTransientTrack pFactory;
	vector<RefCountedKinematicParticle> PhoParticles;

    	float chi = 0.;				      
    	float ndf = 0.;
	float eMassSigma = eMass*1.e-6;

	//loop over all tracks from the conversion	
	for(unsigned convtrk_ite=0; convtrk_ite<gamma.nTracks(); convtrk_ite++){

      	  Track ConvTrk = *(gamma.tracks().at(convtrk_ite));

    	  // get kinematic particles          		   
    	  TransientTrack ConvDau = (*theTTB).build(ConvTrk); 
          PhoParticles.push_back(pFactory.particle(ConvDau, eMass, chi, ndf, eMassSigma));

	}

	KinematicParticleVertexFitter fitter;     

    	if(PhoParticles.size() == 2){

	  RefCountedKinematicTree PhoVertexFitTree = fitter.fit(PhoParticles);

	  if(PhoVertexFitTree->isValid()){

      	    PhoVertexFitTree->movePointerToTheTop();						       
            RefCountedKinematicParticle PhoFitKinematicParticle = PhoVertexFitTree->currentParticle();

	    KinematicState theCurrentKinematicState = PhoFitKinematicParticle->currentState();
            FreeTrajectoryState thePhoFTS = theCurrentKinematicState.freeTrajectoryState();
            genPhoTT = (*theTTB).build(thePhoFTS);

	  }

	}

  	double IpMin = 10.;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);
	        
	  double genPho3DIp = 10001.;
	  double genPho3DIpSig = 10001.;
	  pair<bool,Measurement1D> genPho3DIpPair = IPTools::signedImpactParameter3D(genPhoTT, phoMomentum, *vertexref);

	  if(genPho3DIpPair.first){
    	    genPho3DIp = fabs(genPho3DIpPair.second.value());
	    genPho3DIpSig = fabs(genPho3DIpPair.second.significance());
	  }
 
	  //find and store the closest vertex
	  if(ass_type){
            if(genPho3DIp<IpMin){
              IpMin = genPho3DIp; 
              bestvertexref = vertexref;
            }
	  }else{
            if(genPho3DIpSig<IpMin){
              IpMin = genPho3DIpSig; 
              bestvertexref = vertexref;
            }
	  }

        }

	return make_pair(bestvertexref,make_pair(trackref,-2.));
}


/*************************************************************************************/
/* function to find out if the track comes from a V0 decay                           */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::ComesFromV0Decay(const TrackRef trackref, Handle<VertexCompositeCandidateCollection> vertCompCandCollKshortH, 
	 	 	  	     Handle<VertexCompositeCandidateCollection> vertCompCandCollLambdaH, VertexCompositeCandidate* V0)
{

	//the part for the reassociation of particles from Kshort decays
	for(VertexCompositeCandidateCollection::const_iterator iKS=vertCompCandCollKshortH->begin(); iKS!=vertCompCandCollKshortH->end(); iKS++){

	  const RecoChargedCandidate *dauCand1 = dynamic_cast<const RecoChargedCandidate*>(iKS->daughter(0));
 	  TrackRef dauTk1 = dauCand1->track();
	  const RecoChargedCandidate *dauCand2 = dynamic_cast<const RecoChargedCandidate*>(iKS->daughter(1));
 	  TrackRef dauTk2 = dauCand2->track();

	  if((trackref==dauTk1) || (trackref==dauTk2)){
	  
	    *V0 = *iKS; 
	    return true;

	  }

	}

	//the part for the reassociation of particles from Lambda decays
	for(VertexCompositeCandidateCollection::const_iterator iLambda=vertCompCandCollLambdaH->begin(); iLambda!=vertCompCandCollLambdaH->end(); iLambda++){

	  const RecoChargedCandidate *dauCand1 = dynamic_cast<const RecoChargedCandidate*>(iLambda->daughter(0));
 	  TrackRef dauTk1 = dauCand1->track();
	  const RecoChargedCandidate *dauCand2 = dynamic_cast<const RecoChargedCandidate*>(iLambda->daughter(1));
 	  TrackRef dauTk2 = dauCand2->track();

   	  if((trackref==dauTk1) || (trackref==dauTk2)){
	  
	    *V0 = *iLambda; 
	    return true;

	  }

	}

	return false;
}


/*************************************************************************************/
/* function to find the closest vertex in z for a track from a V0                    */ 
/*************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::FindV0Vertex(const TrackRef trackref, VertexCompositeCandidate V0, Handle<VertexCollection> vtxcollH)
{

        double ztrackfirst = V0.vertex().z();
	double radius = V0.vertex().rho();
	double tracktheta = V0.p4().theta();

	double ztrack = ztrackfirst - (radius/tan(tracktheta));

	return make_pair(PF_PU_AssoMapAlgos::FindClosestInZ(ztrack,vtxcollH),make_pair(trackref,-2.));
}


/*************************************************************************************/
/* function to find out if the track comes from a nuclear interaction                */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::ComesFromNI(const TrackRef trackref, Handle<PFDisplacedVertexCollection> displVertexCollH, PFDisplacedVertex* displVtx)
{

	//the part for the reassociation of particles from nuclear interactions
	for(PFDisplacedVertexCollection::const_iterator iDisplV=displVertexCollH->begin(); iDisplV!=displVertexCollH->end(); iDisplV++){

	  if((iDisplV->isNucl()) && (iDisplV->position().rho()>2.9) && (iDisplV->trackWeight(trackref)>0.)){
	  
	    *displVtx = *iDisplV; 
	    return true;

	  }

	}

	return false;
}


/*************************************************************************************/
/* function to find the closest vertex in z for a track from a nuclear interaction   */ 
/*************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::FindNIVertex(const TrackRef trackref, PFDisplacedVertex displVtx, Handle<VertexCollection> vtxcollH, bool oneDim, const edm::EventSetup& iSetup)
{

	TrackCollection refittedTracks = displVtx.refittedTracks();

	if((displVtx.isTherePrimaryTracks()) || (displVtx.isThereMergedTracks())){

	  for(TrackCollection::const_iterator trkcoll_ite=refittedTracks.begin(); trkcoll_ite!=refittedTracks.end(); trkcoll_ite++){
	
	    const TrackBaseRef retrackbaseref = displVtx.originalTrack(*trkcoll_ite); 

	    if(displVtx.isIncomingTrack(retrackbaseref)){

              VertexTrackQuality VOAssociation = PF_PU_AssoMapAlgos::TrackWeightAssociation(retrackbaseref.castTo<TrackRef>(),vtxcollH);

	      if(VOAssociation.second.second<0.00001){ 
                if(oneDim) VOAssociation = PF_PU_AssoMapAlgos::AssociateClosestInZ(retrackbaseref.castTo<TrackRef>(),vtxcollH);
                else VOAssociation = PF_PU_AssoMapAlgos::AssociateClosest3D(retrackbaseref.castTo<TrackRef>(),vtxcollH,iSetup,false);
	      }

	      return make_pair(VOAssociation.first,make_pair(trackref,-2.));; 

	    }

	  }

	}
	
	math::XYZTLorentzVector mom_sec = displVtx.secondaryMomentum((string) "PI", true);

        double ztrackfirst = displVtx.position().z();
	double radius = displVtx.position().rho();     
	double tracktheta = mom_sec.theta();	

	double ztrack = ztrackfirst - (radius/tan(tracktheta));

	if(oneDim) return make_pair(PF_PU_AssoMapAlgos::FindClosestInZ(ztrack,vtxcollH),make_pair(trackref,-2.));
	else{

	  VertexRef bestvertexref(vtxcollH,0);

	  ESHandle<TransientTrackBuilder> theTTB;
  	  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTB); 

	  ESHandle<MagneticField> bFieldHandle;
	  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);

	  GlobalVector trkMomentum = GlobalVector( mom_sec.x(),
					           mom_sec.y(),
				     	   	   mom_sec.z() );

	  const math::XYZPoint interactionPoint = displVtx.position();
	  const math::XYZVector primaryVector( trkMomentum.x(),trkMomentum.y(),trkMomentum.z() );

	  Track primary = Track(displVtx.chi2(),displVtx.ndof(),interactionPoint,primaryVector,0,trackref->innerStateCovariance());
	  TransientTrack genPhoTT(primary, &(*bFieldHandle) );
    
          KinematicParticleFactoryFromTransientTrack pFactory;
	  vector<RefCountedKinematicParticle> PhoParticles;

    	  float chi = 0.;				      
    	  float ndf = 0.;
	  float piMassSigma = piMass*1.e-6;

	  //loop over all tracks from the nuclear interaction	
	  for(unsigned convtrk_ite=0; convtrk_ite<refittedTracks.size(); convtrk_ite++){

    	    // get kinematic particles          		   
    	    TransientTrack ConvDau = (*theTTB).build(refittedTracks.at(convtrk_ite)); 
            PhoParticles.push_back(pFactory.particle(ConvDau, piMass, chi, ndf, piMassSigma));

	  }

	  KinematicParticleVertexFitter fitter;    

	  RefCountedKinematicTree PhoVertexFitTree = fitter.fit(PhoParticles);

	  if(PhoVertexFitTree->isValid()){

      	    PhoVertexFitTree->movePointerToTheTop();						       
            RefCountedKinematicParticle PhoFitKinematicParticle = PhoVertexFitTree->currentParticle();

	    KinematicState theCurrentKinematicState = PhoFitKinematicParticle->currentState();
            FreeTrajectoryState thePhoFTS = theCurrentKinematicState.freeTrajectoryState();
            genPhoTT = (*theTTB).build(thePhoFTS);

	  }

  	  double IpMin = 10.;
          
	  //loop over all vertices with a good quality in the vertex collection
  	  for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

            VertexRef vertexref(vtxcollH,index_vtx);
	        
	    double genPho3DIpSig = 10001.;
	    pair<bool,Measurement1D> genPho3DIpPair = IPTools::signedImpactParameter3D(genPhoTT, trkMomentum, *vertexref);

	    if(genPho3DIpPair.first){
	      genPho3DIpSig = fabs(genPho3DIpPair.second.significance());
	    }
 
	    //find and store the closest vertex
            if(genPho3DIpSig<IpMin){
              IpMin = genPho3DIpSig; 
              bestvertexref = vertexref;
            }
          }

	return make_pair(bestvertexref,make_pair(trackref,-2.));

        } 

}


/*****************************************************************************************/
/* function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2  */ 
/*****************************************************************************************/

auto_ptr<TrackVertexAssMap>  
PF_PU_AssoMapAlgos::SortAssociationMap(TrackVertexAssMap* trackvertexassInput) 
{
	//create a new TrackVertexAssMap for the Output which will be sorted
     	auto_ptr<TrackVertexAssMap> trackvertexassOutput(new TrackVertexAssMap() );

	//Create and fill a vector of pairs of vertex and the summed (pT-pT_Error)**2 of the tracks associated to the vertex 
	VertexPtsumVector vertexptsumvector;

	//loop over all vertices in the association map
        for(TrackVertexAssMap::const_iterator assomap_ite=trackvertexassInput->begin(); assomap_ite!=trackvertexassInput->end(); assomap_ite++){

	  const VertexRef assomap_vertexref = assomap_ite->key;
  	  const TrackQualityPairVector trckcoll = assomap_ite->val;

	  float ptsum = 0;
 
	  TrackRef trackref;

	  //get the tracks associated to the vertex and calculate the manipulated pT**2
	  for(unsigned int trckcoll_ite=0; trckcoll_ite<trckcoll.size(); trckcoll_ite++){

	    trackref = trckcoll[trckcoll_ite].first;
	    double man_pT = trackref->pt() - trackref->ptError();
	    if(man_pT>0.) ptsum+=man_pT*man_pT;

	  }

	  vertexptsumvector.push_back(make_pair(assomap_vertexref,ptsum));

	}

	while (vertexptsumvector.size()!=0){

	  VertexRef vertexref_highestpT;
	  float highestpT = 0.;
	  int highestpT_index = 0;

	  for(unsigned int vtxptsumvec_ite=0; vtxptsumvec_ite<vertexptsumvector.size(); vtxptsumvec_ite++){
 
 	    if(vertexptsumvector[vtxptsumvec_ite].second>highestpT){

	      vertexref_highestpT = vertexptsumvector[vtxptsumvec_ite].first;
	      highestpT = vertexptsumvector[vtxptsumvec_ite].second;
	      highestpT_index = vtxptsumvec_ite;
	
	    }

	  }
	  
	  //loop over all vertices in the association map
          for(TrackVertexAssMap::const_iterator assomap_ite=trackvertexassInput->begin(); assomap_ite!=trackvertexassInput->end(); assomap_ite++){

	    const VertexRef assomap_vertexref = assomap_ite->key;
  	    const TrackQualityPairVector trckcoll = assomap_ite->val;

	    //if the vertex from the association map the vertex with the highest manipulated pT 
	    //insert all associated tracks in the output Association Map
	    if(assomap_vertexref==vertexref_highestpT) 
	      for(unsigned int trckcoll_ite=0; trckcoll_ite<trckcoll.size(); trckcoll_ite++) 
	        trackvertexassOutput->insert(assomap_vertexref,trckcoll[trckcoll_ite]);
 
	  }

	  vertexptsumvector.erase(vertexptsumvector.begin()+highestpT_index);	

	}

  	return trackvertexassOutput;

}