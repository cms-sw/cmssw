#include "CommonTools/RecoUtils/interface/PF_PU_AssoMapAlgos.h"

#include <vector>
#include <string>
#include <algorithm>

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
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
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
const double kMass = 0.49765;
const double lamMass = 1.11568;
const double piMass = 0.1396;

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;

  typedef pair<TrackRef, float> TrackQualityPair;
  typedef vector< TrackQualityPair > TrackQualityPairVector;
  typedef pair<VertexRef, TrackQualityPair> VertexTrackQuality;

  typedef pair <VertexRef, float>  VertexPtsumPair;
  typedef vector< VertexPtsumPair > VertexPtsumVector;

  typedef math::XYZTLorentzVector LorentzVector;

/*************************************************************************************/
/* dedicated constructor for the algorithms                                          */ 
/*************************************************************************************/

PF_PU_AssoMapAlgos::PF_PU_AssoMapAlgos(const edm::ParameterSet& iConfig)
  : maxNumWarnings_(3),
    numWarnings_(0)
{

  	input_VertexCollection_= iConfig.getParameter<InputTag>("VertexCollection");

  	input_PtCut_ = iConfig.getParameter<double>("TrackPtCut");

  	input_BeamSpot_= iConfig.getParameter<InputTag>("BeamSpot");

  	input_doReassociation_= iConfig.getParameter<bool>("doReassociation");
  	cleanedColls_ = iConfig.getParameter<bool>("GetCleanedCollections");
  
  	ConversionsCollection_= iConfig.getParameter<InputTag>("ConversionsCollection");

  	KshortCollection_= iConfig.getParameter<InputTag>("V0KshortCollection");
  	LambdaCollection_= iConfig.getParameter<InputTag>("V0LambdaCollection");

  	NIVertexCollection_= iConfig.getParameter<InputTag>("NIVertexCollection");

  	UseBeamSpotCompatibility_= iConfig.getUntrackedParameter<bool>("UseBeamSpotCompatibility", false);
  	input_BSCut_ = iConfig.getParameter<double>("BeamSpotCompatibilityCut");

  	input_FinalAssociation_= iConfig.getUntrackedParameter<int>("FinalAssociation", 0);

  	ignoremissingpfcollection_ = iConfig.getParameter<bool>("ignoreMissingCollection");

  	input_nTrack_ = iConfig.getParameter<double>("nTrackWeight");


	/****************************************/
	/* Printing the configuration of the AM */
	/****************************************/

	if(0){	

	  cout << "0. Step: PT-Cut is " << input_PtCut_ << endl;
	  cout << "1. Step: Track weight association" << endl;
	  if ( UseBeamSpotCompatibility_ ){
            cout << "With BSCompatibility check" << endl;
	    goto ending;
	  }else{
            cout << "Without BSCompatibility check" << endl;
	  }
	  if ( input_doReassociation_ ){
            cout << "With Reassociation" << endl;
	  }else{
            cout << "Without Reassociation" << endl;
	  }
	  cout << "The final association is: ";
	  switch (input_FinalAssociation_) {
	  
 	    case 1:{
	      cout << "ClosestInZ" << endl;
	      goto ending;
            }
	  
 	    case 2:{
	      cout << "ClosestIn3D" << endl;
	      goto ending;
            }
	  
 	    default:{
	      cout << "AlwaysFirst" << endl;
	      goto ending;
            }

	  }

	  ending:
	  cout << "" << endl;

	}	

}

/*************************************************************************************/
/* get all needed collections at the beginning                                       */ 
/*************************************************************************************/

bool 
PF_PU_AssoMapAlgos::GetInputCollections(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  	//get the offline beam spot
  	iEvent.getByLabel(input_BeamSpot_, beamspotH);

  	//get the conversion collection for the gamma conversions
  	iEvent.getByLabel(ConversionsCollection_, convCollH);
	cleanedConvCollP = PF_PU_AssoMapAlgos::GetCleanedConversions(convCollH,beamspotH,cleanedColls_);

  	//get the vertex composite candidate collection for the Kshort's
  	iEvent.getByLabel(KshortCollection_, vertCompCandCollKshortH);
	cleanedKshortCollP = PF_PU_AssoMapAlgos::GetCleanedKshort(vertCompCandCollKshortH,beamspotH,cleanedColls_);
  
  	//get the vertex composite candidate collection for the Lambda's
  	iEvent.getByLabel(LambdaCollection_, vertCompCandCollLambdaH);
	cleanedLambdaCollP = PF_PU_AssoMapAlgos::GetCleanedLambda(vertCompCandCollLambdaH,beamspotH,cleanedColls_);
  
  	//get the displaced vertex collection for nuclear interactions
  	//create a new bool, false if no displaced vertex collection is in the event, mostly for AOD
  	missingColls = false;
  	if(!iEvent.getByLabel(NIVertexCollection_,displVertexCollH)){
          if (ignoremissingpfcollection_){

    	    missingColls = true; 

            if ( numWarnings_ < maxNumWarnings_ ) {
	      edm::LogWarning("PF_PU_AssoMap::GetInputCollections")
	        << "No Extra objects available in input file --> skipping reconstruction of photon conversions && displaced vertices !!" << std::endl;
	      ++numWarnings_;
            }

  	  }
	} else {

	    cleanedNICollP = PF_PU_AssoMapAlgos::GetCleanedNI(displVertexCollH,beamspotH,cleanedColls_);

	}
	  
  	//get the input vertex collection
  	iEvent.getByLabel(input_VertexCollection_, vtxcollH);

     	iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);

	//return true if there is at least one reconstructed vertex in the collection
	return (vtxcollH->size()!=0);

}

/*************************************************************************************/
/* do the association for a certain track                                            */ 
/*************************************************************************************/

VertexTrackQuality 
PF_PU_AssoMapAlgos::DoTrackAssociation(const TrackRef& trackref, const edm::EventSetup& iSetup)
{

	VertexTrackQuality VtxTrkQualAss;

	//Step 0:
	//Check for high pt tracks and associate to first vertex
	if ( trackref->pt()>=input_PtCut_ ){
        
	    VtxTrkQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(trackref,  0.));
            return VtxTrkQualAss;

 	}

	// Step 1: First round of association:
    	// Find the vertex with the highest track-to-vertex association weight 
    	VtxTrkQualAss = PF_PU_AssoMapAlgos::TrackWeightAssociation(trackref, vtxcollH);

    	if ( VtxTrkQualAss.second.second >= 1.e-5 ) return VtxTrkQualAss;

	//Step 12: Check for BeamSpot comptibility
	//If a track's vertex is compatible with the BeamSpot
	//look for the closest vertex in z, 
	//if not associate the track always to the first vertex
	if ( UseBeamSpotCompatibility_ ){

          if (PF_PU_AssoMapAlgos::CheckBeamSpotCompability(trackref->vertex(), beamspotH, input_BSCut_) ){
	    VtxTrkQualAss = PF_PU_AssoMapAlgos::AssociateClosestZ(trackref, vtxcollH, input_nTrack_);
	  } else {
 	    VtxTrkQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(trackref, 3.));
          }

	  return VtxTrkQualAss;

        }

	// Step 2: Reassociation
    	// Second round of association:
    	// In case no vertex with track-to-vertex association weight > 1.e-5 is found,
    	// check the track originates from a neutral hadron decay, photon conversion or nuclear interaction

	if (input_doReassociation_) {

      	  // Test if the track comes from a photon conversion:
      	  // If so, try to find the vertex of the mother particle
      	  if ( !missingColls ) {
	    Conversion gamma;
            if ( PF_PU_AssoMapAlgos::ComesFromConversion(trackref, *cleanedConvCollP, &gamma) ){
  	      VtxTrkQualAss = PF_PU_AssoMapAlgos::FindConversionVertex(trackref, gamma, bFieldH, iSetup, beamspotH, vtxcollH, input_nTrack_);
            }
          }

      	  if ( VtxTrkQualAss.second.second == 2. ) return VtxTrkQualAss;

      	  // Test if the track comes from a Kshort or Lambda decay:
      	  // If so, reassociate the track to the vertex of the V0
	  VertexCompositeCandidate V0;
	  if ( PF_PU_AssoMapAlgos::ComesFromV0Decay(trackref, *cleanedKshortCollP, *cleanedLambdaCollP, &V0) ) {
            VtxTrkQualAss = PF_PU_AssoMapAlgos::FindV0Vertex(trackref, V0, bFieldH, iSetup, beamspotH, vtxcollH, input_nTrack_);	
	  }

      	  if ( VtxTrkQualAss.second.second == 2. ) return VtxTrkQualAss;

      	  // Test if the track comes from a nuclear interaction:
      	  // If so, reassociate the track to the vertex of the incoming particle      
      	  if ( !missingColls ) {

	    PFDisplacedVertex displVtx;
	    if ( PF_PU_AssoMapAlgos::ComesFromNI(trackref, *cleanedNICollP, &displVtx) ){
	      VtxTrkQualAss = PF_PU_AssoMapAlgos::FindNIVertex(trackref, displVtx, bFieldH, iSetup, beamspotH, vtxcollH, input_nTrack_);
	    }
	
          }

      	  if ( VtxTrkQualAss.second.second == 2. ) return VtxTrkQualAss;

	}

	// Step 3: Final association
      	// If no vertex is found with track-to-vertex association weight > 1.e-5
      	// and no reassociation was done do the final association 
	// look for the closest vertex in 3D or in z/longitudinal distance
	// or associate the track always to the first vertex (default)

	switch (input_FinalAssociation_) {
	  
 	  case 1:{
            return PF_PU_AssoMapAlgos::AssociateClosestZ(trackref, vtxcollH, input_nTrack_);
          }
	  
 	  case 2:{
            return PF_PU_AssoMapAlgos::AssociateClosest3D(trackref, vtxcollH, bFieldH, iSetup, beamspotH, input_nTrack_);
          }
	  
 	  default:{
            return make_pair(VertexRef(vtxcollH, 0), make_pair(trackref, 3.));
          }

	}

}

/*************************************************************************************/
/* returns the first vertex of the vertex collection                                 */ 
/*************************************************************************************/

VertexRef  
PF_PU_AssoMapAlgos::GetFirstVertex() 
{

	VertexRef vtxref_tmp(vtxcollH,0);

	return vtxref_tmp;

}

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


/*******************************************************************************************/
/* function to associate the track to the closest vertex in z/longitudinal distance        */ 
/*******************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::AssociateClosestZ(TrackRef trackref, Handle<VertexCollection> vtxcollH, double tWeight)
{

	double ztrack = trackref->vertex().z();

	VertexRef bestvertexref(vtxcollH, 0);

	double dzmin = 5.;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);

	  int nTracks = sqrt(vertexref->tracksSize());

          double z_distance = fabs(ztrack - vertexref->z());
	  double dz = max(0.0, z_distance-tWeight*nTracks);	

          if(dz<dzmin) {
            dzmin = dz; 
            bestvertexref = vertexref;
          }
	
	}	

	return make_pair(bestvertexref, make_pair(trackref, 3.));
}


/*************************************************************************************/
/* function to find the closest vertex in 3D for a certain track                     */ 
/*************************************************************************************/

VertexRef 
PF_PU_AssoMapAlgos::FindClosest3D(TransientTrack transtrk, Handle<VertexCollection> vtxcollH, double tWeight)
{

	VertexRef foundVertexRef(vtxcollH, 0);

	double d3min = 5.;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH, index_vtx);

	  GlobalPoint vtxpos(vertexref->x(),vertexref->y(),vertexref->z());
	  GlobalPoint closestPoint = transtrk.trajectoryStateClosestToPoint(vtxpos).position();

	  int nTracks = sqrt(vertexref->tracksSize());
 
	  //find and store the closest vertex in z
	  double x_dist = vtxpos.x() - closestPoint.x();
	  double y_dist = vtxpos.y() - closestPoint.y();
	  double z_dist = vtxpos.z() - closestPoint.z();

          double distance = sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);

	  double weightedDistance = max(0.0, distance-tWeight*nTracks);	

          if(weightedDistance<d3min) {
            d3min = weightedDistance; 
            foundVertexRef = vertexref;
          }
	
	}

	if(d3min<5.) return foundVertexRef;
	  else return VertexRef(vtxcollH, 0);
}


/*******************************************************************************************/
/* function to associate the track to the closest vertex in 3D                             */ 
/*******************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::AssociateClosest3D(TrackRef trackref, Handle<VertexCollection> vtxcollH, ESHandle<MagneticField> bFieldH, const EventSetup& iSetup, Handle<BeamSpot> bsH, double tWeight)
{

	TransientTrack transtrk(trackref, &(*bFieldH) );
    	transtrk.setBeamSpot(*bsH);
    	transtrk.setES(iSetup);

	VertexRef bestvertexref = FindClosest3D(transtrk, vtxcollH, tWeight);	

	return make_pair(bestvertexref, make_pair(trackref, 3.));
}


/*************************************************************************************/
/* function to filter the conversion collection                                      */ 
/*************************************************************************************/

auto_ptr<ConversionCollection> 
PF_PU_AssoMapAlgos::GetCleanedConversions(edm::Handle<reco::ConversionCollection> convCollH, Handle<BeamSpot> bsH, bool cleanedColl)
{
     	auto_ptr<ConversionCollection> cleanedConvColl(new ConversionCollection() );

	for (unsigned int convcoll_idx=0; convcoll_idx<convCollH->size(); convcoll_idx++){

	  ConversionRef convref(convCollH,convcoll_idx);

 	  if(!cleanedColl){   
            cleanedConvColl->push_back(*convref);
	    continue;
          }

	  if( (convref->quality(Conversion::ConversionQuality(8))) &&
              (convref->nTracks()==2) &&
              (convref->dxy()>3.) &&
              (fabs(convref->pairInvariantMass())<=5.) ){

	    double connVec_x = convref->conversionVertex().x() - bsH->x0();
	    double connVec_y = convref->conversionVertex().y() - bsH->y0();
	    double connVec_z = convref->conversionVertex().z() - bsH->z0();

	    double connVec_r = sqrt(connVec_x*connVec_x + connVec_y*connVec_y + connVec_z*connVec_z);

	    double connVec_theta = acos(connVec_z*1./connVec_r);
	    double connVec_eta = -1.*log(tan(connVec_theta*1./2.));
	    double connVec_phi = atan2(connVec_y,connVec_x);

	    double incom_eta = convref->refittedPair4Momentum().eta();
	    double incom_phi = convref->refittedPair4Momentum().phi();

	    if ( deltaR(incom_eta,incom_phi,connVec_eta,connVec_phi)<=0.3 ){    
              cleanedConvColl->push_back(*convref);
            }

	  }

	}

  	return cleanedConvColl;

}


/*************************************************************************************/
/* function to find out if the track comes from a gamma conversion                   */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::ComesFromConversion(const TrackRef trackref, ConversionCollection cleanedConvColl, Conversion* gamma)
{

	for(unsigned int convcoll_ite=0; convcoll_ite<cleanedConvColl.size(); convcoll_ite++){

	  if(ConversionTools::matchesConversion(trackref,cleanedConvColl.at(convcoll_ite))){
	
	    *gamma = cleanedConvColl.at(convcoll_ite);
	    return true;

  	  }

  	}

	return false;
}


/*************************************************************************************/
/* function to find the closest vertex in z for a track from a conversion            */ 
/*************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::FindConversionVertex(const reco::TrackRef trackref, reco::Conversion gamma, ESHandle<MagneticField> bFieldH, const EventSetup& iSetup, edm::Handle<reco::BeamSpot> bsH, edm::Handle<reco::VertexCollection> vtxcollH, double tWeight)
{ 

	math::XYZPoint conv_pos = gamma.conversionVertex().position();

	math::XYZVector conv_mom(gamma.refittedPair4Momentum().x(),
	                         gamma.refittedPair4Momentum().y(),
	                         gamma.refittedPair4Momentum().z());

	Track photon(trackref->chi2(), trackref->ndof(), conv_pos, conv_mom, 0, trackref->covariance());

    	TransientTrack transpho(photon, &(*bFieldH) );
    	transpho.setBeamSpot(*bsH);
    	transpho.setES(iSetup);

	VertexRef foundVertexRef = FindClosest3D(transpho, vtxcollH, tWeight); 

	return make_pair(foundVertexRef, make_pair(trackref, 2.));

}


/*************************************************************************************/
/* function to filter the Kshort collection                                          */ 
/*************************************************************************************/

auto_ptr<VertexCompositeCandidateCollection>
PF_PU_AssoMapAlgos::GetCleanedKshort(Handle<VertexCompositeCandidateCollection> KshortsH, Handle<BeamSpot> bsH, bool cleanedColl)
{

     	auto_ptr<VertexCompositeCandidateCollection> cleanedKshortColl(new VertexCompositeCandidateCollection() );

	for (unsigned int kshortcoll_idx=0; kshortcoll_idx<KshortsH->size(); kshortcoll_idx++){

	  VertexCompositeCandidateRef kshortref(KshortsH,kshortcoll_idx);

 	  if(!cleanedColl){   
            cleanedKshortColl->push_back(*kshortref);
	    continue;
          }

     	  double kschi2 = kshortref->vertexNormalizedChi2();

	  if( (kshortref->vertex().rho()>=3.) && 
 	      (kschi2 < 3.) &&
	      (fabs(kshortref->mass() - kMass)<=0.01) ){

	    double kVert_x = kshortref->vertex().x();
	    double kVert_xErr2 = kshortref->vertexCovariance(0,0)*kshortref->vertexCovariance(0,0);
	    double kVert_y = kshortref->vertex().y();
	    double kVert_yErr2 = kshortref->vertexCovariance(1,1)*kshortref->vertexCovariance(1,1);
	    double kVert_z = kshortref->vertex().z();
	    double kVert_zErr2 = kshortref->vertexCovariance(2,2)*kshortref->vertexCovariance(2,2);

	    double bs_x = bsH->x0();
	    double bs_xErr2 = bsH->x0Error()*bsH->x0Error();
	    double bs_y = bsH->y0();
	    double bs_yErr2 = bsH->y0Error()*bsH->y0Error();
	    double bs_z = bsH->z0();
	    double bs_zErr2 = bsH->z0Error()*bsH->z0Error();

	    double x_distance2 = (kVert_x - bs_x)*(kVert_x - bs_x);
	    double y_distance2 = (kVert_y - bs_y)*(kVert_y - bs_y);
	    double z_distance2 = (kVert_z - bs_z)*(kVert_z - bs_z); 

	    double distance = sqrt(x_distance2+y_distance2+z_distance2);
	    double error = sqrt(kVert_xErr2+kVert_yErr2+kVert_zErr2+bs_xErr2+bs_yErr2+bs_zErr2);

	    if (distance>=5.*error){

	      double connVec_x = kVert_x - bs_x;
	      double connVec_y = kVert_y - bs_x;
	      double connVec_z = kVert_z - bs_x;

	      double connVec_r = sqrt(connVec_x*connVec_x + connVec_y*connVec_y + connVec_z*connVec_z);

	      double connVec_theta = acos(connVec_z*1./connVec_r);
	      double connVec_eta = -1.*log(tan(connVec_theta*1./2.));
	      double connVec_phi = atan2(connVec_y,connVec_x);

	      double incom_eta = kshortref->momentum().eta();
	      double incom_phi = kshortref->momentum().phi();

	      if ( deltaR(incom_eta,incom_phi,connVec_eta,connVec_phi)<=0.3 )    
                cleanedKshortColl->push_back(*kshortref);

	    }

	  }

	}

	return cleanedKshortColl;

}


/*************************************************************************************/
/* function to filter the Lambda collection                                          */ 
/*************************************************************************************/

auto_ptr<VertexCompositeCandidateCollection>
PF_PU_AssoMapAlgos::GetCleanedLambda(Handle<VertexCompositeCandidateCollection> LambdasH, Handle<BeamSpot> bsH, bool cleanedColl)
{

     	auto_ptr<VertexCompositeCandidateCollection> cleanedLambdaColl(new VertexCompositeCandidateCollection() );

	for (unsigned int lambdacoll_idx=0; lambdacoll_idx<LambdasH->size(); lambdacoll_idx++){

	  VertexCompositeCandidateRef lambdaref(LambdasH,lambdacoll_idx);

 	  if(!cleanedColl){   
            cleanedLambdaColl->push_back(*lambdaref);
	    continue;
          }

     	  double lamchi2 = lambdaref->vertexNormalizedChi2();

	  if( (lambdaref->vertex().rho()>=3.) && 
 	      (lamchi2 < 3.) &&
	      (fabs(lambdaref->mass() - lamMass)<=0.005) ){

	    double lVert_x = lambdaref->vertex().x();
	    double lVert_xErr2 = lambdaref->vertexCovariance(0,0)*lambdaref->vertexCovariance(0,0);
	    double lVert_y = lambdaref->vertex().y();
	    double lVert_yErr2 = lambdaref->vertexCovariance(1,1)*lambdaref->vertexCovariance(1,1);
	    double lVert_z = lambdaref->vertex().z();
	    double lVert_zErr2 = lambdaref->vertexCovariance(2,2)*lambdaref->vertexCovariance(2,2);

	    double bs_x = bsH->x0();
	    double bs_xErr2 = bsH->x0Error()*bsH->x0Error();
	    double bs_y = bsH->y0();
	    double bs_yErr2 = bsH->y0Error()*bsH->y0Error();
	    double bs_z = bsH->z0();
	    double bs_zErr2 = bsH->z0Error()*bsH->z0Error();

	    double x_distance2 = (lVert_x - bs_x)*(lVert_x - bs_x);
	    double y_distance2 = (lVert_y - bs_y)*(lVert_y - bs_y);
	    double z_distance2 = (lVert_z - bs_z)*(lVert_z - bs_z); 

	    double distance = sqrt(x_distance2+y_distance2+z_distance2);
	    double error = sqrt(lVert_xErr2+lVert_yErr2+lVert_zErr2+bs_xErr2+bs_yErr2+bs_zErr2);

	    if (distance>=10.*error){

	      double connVec_x = lVert_x - bs_x;
	      double connVec_y = lVert_y - bs_x;
	      double connVec_z = lVert_z - bs_x;

	      double connVec_r = sqrt(connVec_x*connVec_x + connVec_y*connVec_y + connVec_z*connVec_z);

	      double connVec_theta = acos(connVec_z*1./connVec_r);
	      double connVec_eta = -1.*log(tan(connVec_theta*1./2.));
	      double connVec_phi = atan2(connVec_y,connVec_x);

	      double incom_eta = lambdaref->momentum().eta();
	      double incom_phi = lambdaref->momentum().phi();

	      if ( deltaR(incom_eta,incom_phi,connVec_eta,connVec_phi)<=0.3 )    
                cleanedLambdaColl->push_back(*lambdaref);

	    }

	  }

	}

	return cleanedLambdaColl;
}

/*************************************************************************************/
/* function to find out if the track comes from a V0 decay                           */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::ComesFromV0Decay(const TrackRef trackref, VertexCompositeCandidateCollection cleanedKshort, VertexCompositeCandidateCollection cleanedLambda, VertexCompositeCandidate* V0)
{

	//the part for the reassociation of particles from Kshort decays
	for(VertexCompositeCandidateCollection::const_iterator iKS=cleanedKshort.begin(); iKS!=cleanedKshort.end(); iKS++){

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
	for(VertexCompositeCandidateCollection::const_iterator iLambda=cleanedLambda.begin(); iLambda!=cleanedLambda.end(); iLambda++){

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
PF_PU_AssoMapAlgos::FindV0Vertex(const TrackRef trackref, VertexCompositeCandidate V0_vtx, ESHandle<MagneticField> bFieldH, const EventSetup& iSetup, Handle<BeamSpot> bsH, Handle<VertexCollection> vtxcollH, double tWeight)
{ 

	math::XYZPoint dec_pos = V0_vtx.vertex();

	math::XYZVector dec_mom(V0_vtx.momentum().x(),
	                        V0_vtx.momentum().y(),
	                        V0_vtx.momentum().z());

	Track V0(trackref->chi2(), trackref->ndof(), dec_pos, dec_mom, 0, trackref->covariance());

    	TransientTrack transV0(V0, &(*bFieldH) );
    	transV0.setBeamSpot(*bsH);
    	transV0.setES(iSetup);

	VertexRef foundVertexRef = FindClosest3D(transV0, vtxcollH, tWeight); 

	return make_pair(foundVertexRef, make_pair(trackref, 2.));
}


/*************************************************************************************/
/* function to filter the nuclear interaction collection                             */ 
/*************************************************************************************/

auto_ptr<PFDisplacedVertexCollection>
PF_PU_AssoMapAlgos::GetCleanedNI(Handle<PFDisplacedVertexCollection> NuclIntH, Handle<BeamSpot> bsH, bool cleanedColl)
{

     	auto_ptr<PFDisplacedVertexCollection> cleanedNIColl(new PFDisplacedVertexCollection() );

	for (PFDisplacedVertexCollection::const_iterator niref=NuclIntH->begin(); niref!=NuclIntH->end(); niref++){

	  if(!cleanedColl){
	    cleanedNIColl->push_back(*niref);
	    continue;
          }

	  if( !(niref->isFake()) && 
 	      (niref->position().rho()>3.) &&
	      (niref->isNucl()) ){

	    double niVert_x = niref->x();
	    double niVert_xErr2 = niref->xError()*niref->xError();
	    double niVert_y = niref->y();
	    double niVert_yErr2 = niref->yError()*niref->yError();
	    double niVert_z = niref->z();
	    double niVert_zErr2 = niref->zError()*niref->zError();

	    double bs_x = bsH->x0();
	    double bs_xErr2 = bsH->x0Error()*bsH->x0Error();
	    double bs_y = bsH->y0();
	    double bs_yErr2 = bsH->y0Error()*bsH->y0Error();
	    double bs_z = bsH->z0();
	    double bs_zErr2 = bsH->z0Error()*bsH->z0Error();

	    double x_distance2 = (niVert_x - bs_x)*(niVert_x - bs_x);
	    double y_distance2 = (niVert_y - bs_y)*(niVert_y - bs_y);
	    double z_distance2 = (niVert_z - bs_z)*(niVert_z - bs_z); 

	    double distance = sqrt(x_distance2+y_distance2+z_distance2);
	    double error = sqrt(niVert_xErr2+niVert_yErr2+niVert_zErr2+bs_xErr2+bs_yErr2+bs_zErr2);

	    if (distance>=5.*error){

	      double connVec_x = niVert_x - bs_x;
	      double connVec_y = niVert_y - bs_x;
	      double connVec_z = niVert_z - bs_x;

	      double connVec_r = sqrt(connVec_x*connVec_x + connVec_y*connVec_y + connVec_z*connVec_z);

	      double connVec_theta = acos(connVec_z*1./connVec_r);
	      double connVec_eta = -1.*log(tan(connVec_theta*1./2.));
	      double connVec_phi = atan2(connVec_y,connVec_x);

	      double incom_eta = niref->primaryMomentum().eta();
	      double incom_phi = niref->primaryMomentum().phi();

	      if ( deltaR(incom_eta,incom_phi,connVec_eta,connVec_phi)<=0.3 )    
                cleanedNIColl->push_back(*niref);

	    }	    

	  }

	}

	return cleanedNIColl;
}


/*************************************************************************************/
/* function to find out if the track comes from a nuclear interaction                */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::ComesFromNI(const TrackRef trackref, PFDisplacedVertexCollection cleanedNI, PFDisplacedVertex* displVtx)
{

	//the part for the reassociation of particles from nuclear interactions
	for(PFDisplacedVertexCollection::const_iterator iDisplV=cleanedNI.begin(); iDisplV!=cleanedNI.end(); iDisplV++){

	  if(iDisplV->trackWeight(trackref)>1.e-2){
	  
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
PF_PU_AssoMapAlgos::FindNIVertex(const TrackRef trackref, PFDisplacedVertex displVtx, ESHandle<MagneticField> bFieldH, const EventSetup& iSetup, Handle<BeamSpot> bsH, Handle<VertexCollection> vtxcollH, double tWeight)
{

	TrackCollection refittedTracks = displVtx.refittedTracks();

	if((displVtx.isTherePrimaryTracks()) || (displVtx.isThereMergedTracks())){

	  for(TrackCollection::const_iterator trkcoll_ite=refittedTracks.begin(); trkcoll_ite!=refittedTracks.end(); trkcoll_ite++){
	
	    const TrackBaseRef retrackbaseref = displVtx.originalTrack(*trkcoll_ite); 

	    if(displVtx.isIncomingTrack(retrackbaseref)){

              VertexTrackQuality VOAssociation = PF_PU_AssoMapAlgos::TrackWeightAssociation(retrackbaseref.castTo<TrackRef>(), vtxcollH);

	      if(VOAssociation.second.second>1.e-5) 
                return make_pair(VOAssociation.first, make_pair(trackref, 2.));

    	      TransientTrack transIncom(*retrackbaseref, &(*bFieldH) );
    	      transIncom.setBeamSpot(*bsH);
    	      transIncom.setES(iSetup);

	      VertexRef foundVertexRef = FindClosest3D(transIncom, vtxcollH, tWeight); 

	      return make_pair(foundVertexRef, make_pair(trackref, 2.));

	    }

	  }

	}

	math::XYZPoint ni_pos = displVtx.position();

	math::XYZVector ni_mom(displVtx.primaryMomentum().x(),
	                       displVtx.primaryMomentum().y(),
	                       displVtx.primaryMomentum().z());

	Track incom(trackref->chi2(), trackref->ndof(), ni_pos, ni_mom, 0, trackref->covariance());

    	TransientTrack transIncom(incom, &(*bFieldH) );
    	transIncom.setBeamSpot(*bsH);
    	transIncom.setES(iSetup);

	VertexRef foundVertexRef = FindClosest3D(transIncom, vtxcollH, tWeight); 

	return make_pair(foundVertexRef, make_pair(trackref, 2.));

}


/*************************************************************************************/
/* function to check if a candidate is compatible with the BeamSpot                  */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::CheckBeamSpotCompability(const math::XYZPoint vtx, Handle<BeamSpot> beamspotH, double cut)
{

        double vtx_x = vtx.x();
        double vtx_y = vtx.y(); 

        double bs_x = beamspotH->x(vtx.z());
        double bs_y = beamspotH->y(vtx.z());

	double relative_x = (vtx_x - bs_x) /  beamspotH->BeamWidthX();
	double relative_y = (vtx_y - bs_y) /  beamspotH->BeamWidthY();

	double relative_distance = sqrt(relative_x*relative_x + relative_y*relative_y);

	return (relative_distance<=cut);

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