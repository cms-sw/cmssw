#include "CommonTools/RecoUtils/interface/PF_PU_AssoMapAlgos.h"

#include <vector>
#include <string>
#include <sstream> 
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

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"


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

const double kMass = 0.49765;
const double lamMass = 1.11568;

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


	string logOutput;
 	ostringstream convert;	

	logOutput+= "0. Step: PT-Cut is ";
        convert << input_PtCut_;	
        logOutput+= convert.str();

        logOutput+= "\n1. Step: Track weight association\n";

 	if ( UseBeamSpotCompatibility_ ){
          logOutput+= "With BSCompatibility check\n";
	  goto ending;
	}else{
          logOutput+= "Without BSCompatibility check\n";
	}
	if ( input_doReassociation_ ){
          logOutput+= "With Reassociation\n";
	}else{
          logOutput+= "Without Reassociation\n";
	}
	logOutput+= "The final association is: ";
	switch (input_FinalAssociation_) {
	
 	  case 1:{
	    logOutput+= "ClosestInZ with weight ";
            convert << input_nTrack_;
            logOutput+= convert.str();
            logOutput+= "\n";
	    goto ending;
          }
	  
 	  case 2:{
	    logOutput+= "ClosestIn3D with weight ";
            convert << input_nTrack_;
            logOutput+= convert.str();
            logOutput+= "\n";
	    goto ending;
          }
	  
 	  default:{
	    logOutput+= "AlwaysFirst\n";
	    goto ending;
          }

	}

	ending:
	logOutput+="\n";

	LogInfo("PF_PU_AssoMap::PF_PU_AssoMapAlgos")
	  << logOutput << endl;	

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

	    cleanedNICollP = PF_PU_AssoMapAlgos::GetCleanedNI(displVertexCollH,beamspotH,true);

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

	TransientTrack transtrk(trackref, &(*bFieldH) );
    	transtrk.setBeamSpot(*beamspotH);
    	transtrk.setES(iSetup);

	VertexTrackQuality VtxTrkQualAss;

	//Step 0:
	//Check for high pt tracks and associate to first vertex
	if ( trackref->pt()>=input_PtCut_ ){
        
            pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *(VertexRef(vtxcollH, 0)) );
 	    VtxTrkQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(trackref, IpPair.second.value()));
            return VtxTrkQualAss;

 	}

	// Step 1: First round of association:
    	// Find the vertex with the highest track-to-vertex association weight 
    	VtxTrkQualAss = PF_PU_AssoMapAlgos::TrackWeightAssociation(trackref, vtxcollH);

    	if ( VtxTrkQualAss.second.second == 0. ) return VtxTrkQualAss;

	//Step 1/2: Check for BeamSpot comptibility
	//If a track's vertex is compatible with the BeamSpot
	//look for the closest vertex in z, 
	//if not associate the track always to the first vertex
	if ( UseBeamSpotCompatibility_ ){

          if (PF_PU_AssoMapAlgos::CheckBeamSpotCompability(transtrk, input_BSCut_) ){
	    VtxTrkQualAss = PF_PU_AssoMapAlgos::AssociateClosestZ(trackref, vtxcollH, input_nTrack_);
	  } else {
            pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *(VertexRef(vtxcollH, 0)) );
 	    VtxTrkQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(trackref, IpPair.second.value()));
          }

	  return VtxTrkQualAss;

        }

	// Step 2: Reassociation
    	// Second round of association:
    	// In case no vertex with track-to-vertex association weight > 1.e-5 is found,
    	// check the track originates from a neutral hadron decay, photon conversion or nuclear interaction

	if ((input_doReassociation_) && (!missingColls)) {

      	  // Test if the track comes from a photon conversion:
      	  // If so, try to find the vertex of the mother particle
	  Conversion gamma;
          if ( PF_PU_AssoMapAlgos::ComesFromConversion(trackref, *cleanedConvCollP, &gamma) ){
  	    VtxTrkQualAss = PF_PU_AssoMapAlgos::FindConversionVertex(trackref, gamma, bFieldH, iSetup, beamspotH, vtxcollH, input_nTrack_);
	    return VtxTrkQualAss;
          }

      	  // Test if the track comes from a Kshort or Lambda decay:
      	  // If so, reassociate the track to the vertex of the V0
	  VertexCompositeCandidate V0;
	  if ( PF_PU_AssoMapAlgos::ComesFromV0Decay(trackref, *cleanedKshortCollP, *cleanedLambdaCollP, &V0) ) {
            VtxTrkQualAss = PF_PU_AssoMapAlgos::FindV0Vertex(trackref, V0, bFieldH, iSetup, beamspotH, vtxcollH, input_nTrack_);	
	    return VtxTrkQualAss;
	  }

      	  // Test if the track comes from a nuclear interaction:
      	  // If so, reassociate the track to the vertex of the incoming particle 
	  PFDisplacedVertex displVtx;
	  if ( PF_PU_AssoMapAlgos::ComesFromNI(trackref, *cleanedNICollP, &displVtx) ){
	    VtxTrkQualAss = PF_PU_AssoMapAlgos::FindNIVertex(trackref, displVtx, bFieldH, iSetup, beamspotH, vtxcollH, input_nTrack_);
	    return VtxTrkQualAss;
	  }

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
            pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *(VertexRef(vtxcollH, 0)) );
 	    VtxTrkQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(trackref, IpPair.second.value()));
          }

	}

	return VtxTrkQualAss;

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

/****************************************************************************************/
/* function to calculate the deltaR between a vector and a vector connecting two points */ 
/****************************************************************************************/

double
PF_PU_AssoMapAlgos::dR(math::XYZPoint vtx_pos, math::XYZVector vtx_mom, edm::Handle<reco::BeamSpot> bsH)
{

	double bs_x = bsH->x0();
	double bs_y = bsH->y0();
	double bs_z = bsH->z0();

     	double connVec_x = vtx_pos.x() - bs_x;
	double connVec_y = vtx_pos.y() - bs_y;
	double connVec_z = vtx_pos.z() - bs_z;

     	double connVec_r = sqrt(connVec_x*connVec_x + connVec_y*connVec_y + connVec_z*connVec_z);
	double connVec_theta = acos(connVec_z*1./connVec_r);

	double connVec_eta = -1.*log(tan(connVec_theta*1./2.));
	double connVec_phi = atan2(connVec_y,connVec_x);

	return deltaR(vtx_mom.eta(),vtx_mom.phi(),connVec_eta,connVec_phi);
    
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

	if ( bestweight>1.e-5 ){ 
	  //found a vertex with a track weight
	  //return weight == 0., so that all following steps won't be applied
  	  return make_pair(bestvertexref,make_pair(trackRef,0.));
	} else { 
	  //found no vertex with a track weight
	  //return weight == 1., so that secondary and final association will be applied
  	  return make_pair(bestvertexref,make_pair(trackRef,1.));
	}

}


/*******************************************************************************************/
/* function to associate the track to the closest vertex in z/longitudinal distance        */ 
/*******************************************************************************************/

VertexTrackQuality
PF_PU_AssoMapAlgos::AssociateClosestZ(TrackRef trackref, Handle<VertexCollection> vtxcollH, double tWeight)
{

	double ztrack = trackref->vertex().z();

	VertexRef bestvertexref(vtxcollH, 0);

	double dzmin = 1e5;
	double realDistance = 1e5;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);

	  int nTracks = vertexref->tracksSize();

          double z_distance = fabs(ztrack - vertexref->z());
	  double dz = z_distance-tWeight*nTracks;	

          if(dz<dzmin) {
            dzmin = dz; 
            realDistance = z_distance; 
            bestvertexref = vertexref;
          }
	
	}	

	return make_pair(bestvertexref, make_pair(trackref, realDistance));
}


/*************************************************************************************/
/* function to find the closest vertex in 3D for a certain track                     */ 
/*************************************************************************************/

VertexRef 
PF_PU_AssoMapAlgos::FindClosest3D(TransientTrack transtrk, Handle<VertexCollection> vtxcollH, double tWeight)
{

	VertexRef foundVertexRef(vtxcollH, 0);

	double d3min = 1e5;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);

	  double nTracks = sqrt(vertexref->tracksSize());

          double distance = 1e5;	        
          pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *vertexref);
 
	  if(IpPair.first)
            distance = IpPair.second.value();

	  double weightedDistance = distance-tWeight*nTracks;	

          if(weightedDistance<d3min) {
            d3min = weightedDistance; 
            foundVertexRef = vertexref;
          }
	
	}

	return foundVertexRef;
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

        pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *bestvertexref);	

	return make_pair(bestvertexref, make_pair(trackref, IpPair.second.value()));
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

	  if( (convref->nTracks()==2) &&
              (fabs(convref->pairInvariantMass())<=0.1) ){
    
            cleanedConvColl->push_back(*convref);

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

	TransientTrack transtrk(trackref, &(*bFieldH) );
    	transtrk.setBeamSpot(*bsH);
    	transtrk.setES(iSetup);

        pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *foundVertexRef);	

	return make_pair(foundVertexRef, make_pair(trackref, -1.*IpPair.second.value()));

}


/*************************************************************************************/
/* function to filter the Kshort collection                                          */ 
/*************************************************************************************/

auto_ptr<VertexCompositeCandidateCollection>
PF_PU_AssoMapAlgos::GetCleanedKshort(Handle<VertexCompositeCandidateCollection> KshortsH, Handle<BeamSpot> bsH, bool cleanedColl)
{

     	auto_ptr<VertexCompositeCandidateCollection> cleanedKaonColl(new VertexCompositeCandidateCollection() );

	for (unsigned int kscoll_idx=0; kscoll_idx<KshortsH->size(); kscoll_idx++){

	  VertexCompositeCandidateRef ksref(KshortsH,kscoll_idx);

 	  if(!cleanedColl){   
            cleanedKaonColl->push_back(*ksref);
	    continue;
	  }

  	  VertexDistance3D distanceComputer;

          GlobalPoint dec_pos = RecoVertex::convertPos(ksref->vertex());    

       	  GlobalError decayVertexError = GlobalError(ksref->vertexCovariance(0,0), ksref->vertexCovariance(0,1), ksref->vertexCovariance(1,1), ksref->vertexCovariance(0,2), ksref->vertexCovariance(1,2), ksref->vertexCovariance(2,2));
	
      	  math::XYZVector dec_mom(ksref->momentum().x(),
	                          ksref->momentum().y(),
	                          ksref->momentum().z());    

      	  GlobalPoint bsPosition = RecoVertex::convertPos(bsH->position());
      	  GlobalError bsError = RecoVertex::convertError(bsH->covariance3D());
      
   	  double kaon_significance = (distanceComputer.distance(VertexState(bsPosition,bsError), VertexState(dec_pos, decayVertexError))).significance();

	  if ((ksref->vertex().rho()>=3.) &&
              (ksref->vertexNormalizedChi2()<=3.) &&
              (fabs(ksref->mass() - kMass)<=0.01) &&
              (kaon_significance>15.) &&
              (PF_PU_AssoMapAlgos::dR(ksref->vertex(),dec_mom,bsH)<=0.3) ){
  
            cleanedKaonColl->push_back(*ksref);

       	  }

	}

	return cleanedKaonColl;

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

  	  VertexDistance3D distanceComputer;

          GlobalPoint dec_pos = RecoVertex::convertPos(lambdaref->vertex());    

       	  GlobalError decayVertexError = GlobalError(lambdaref->vertexCovariance(0,0), lambdaref->vertexCovariance(0,1), lambdaref->vertexCovariance(1,1), lambdaref->vertexCovariance(0,2), lambdaref->vertexCovariance(1,2), lambdaref->vertexCovariance(2,2));
	
      	  math::XYZVector dec_mom(lambdaref->momentum().x(),
	                          lambdaref->momentum().y(),
	                          lambdaref->momentum().z());    

      	  GlobalPoint bsPosition = RecoVertex::convertPos(bsH->position());
      	  GlobalError bsError = RecoVertex::convertError(bsH->covariance3D());
      
   	  double lambda_significance = (distanceComputer.distance(VertexState(bsPosition,bsError), VertexState(dec_pos, decayVertexError))).significance();

	  if ((lambdaref->vertex().rho()>=3.) &&
              (lambdaref->vertexNormalizedChi2()<=3.) &&
              (fabs(lambdaref->mass() - lamMass)<=0.005) &&
              (lambda_significance>15.) &&
              (PF_PU_AssoMapAlgos::dR(lambdaref->vertex(),dec_mom,bsH)<=0.3) ){
  
            cleanedLambdaColl->push_back(*lambdaref);

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

	TransientTrack transtrk(trackref, &(*bFieldH) );
    	transtrk.setBeamSpot(*bsH);
    	transtrk.setES(iSetup);

        pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *foundVertexRef);	

	return make_pair(foundVertexRef, make_pair(trackref, -1.*IpPair.second.value()));

}


/*************************************************************************************/
/* function to filter the nuclear interaction collection                             */ 
/*************************************************************************************/

auto_ptr<PFDisplacedVertexCollection>
PF_PU_AssoMapAlgos::GetCleanedNI(Handle<PFDisplacedVertexCollection> NuclIntH, Handle<BeamSpot> bsH, bool cleanedColl)
{

     	auto_ptr<PFDisplacedVertexCollection> cleanedNIColl(new PFDisplacedVertexCollection() );

	for (PFDisplacedVertexCollection::const_iterator niref=NuclIntH->begin(); niref!=NuclIntH->end(); niref++){


	  if( (niref->isFake()) || !(niref->isNucl()) ) continue;

	  if(!cleanedColl){
	    cleanedNIColl->push_back(*niref);
	    continue;
          }

  	  VertexDistance3D distanceComputer;

      	  GlobalPoint ni_pos = RecoVertex::convertPos(niref->position());    
      	  GlobalError interactionVertexError = RecoVertex::convertError(niref->error());

      	  math::XYZVector ni_mom(niref->primaryMomentum().x(),
	                         niref->primaryMomentum().y(),
	                         niref->primaryMomentum().z());

      	  GlobalPoint bsPosition = RecoVertex::convertPos(bsH->position());
      	  GlobalError bsError = RecoVertex::convertError(bsH->covariance3D());
      
   	  double nuclint_significance = (distanceComputer.distance(VertexState(bsPosition,bsError), VertexState(ni_pos, interactionVertexError))).significance();

	  if ((niref->position().rho()>=3.) &&
              (nuclint_significance>15.) &&
              (PF_PU_AssoMapAlgos::dR(niref->position(),ni_mom,bsH)<=0.3) ){
  
            cleanedNIColl->push_back(*niref);

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

	  if(iDisplV->trackWeight(trackref)>1.e-5){
	  
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

	TransientTrack transtrk(trackref, &(*bFieldH) );
    	transtrk.setBeamSpot(*bsH);
    	transtrk.setES(iSetup);

	TrackCollection refittedTracks = displVtx.refittedTracks();

	if((displVtx.isTherePrimaryTracks()) || (displVtx.isThereMergedTracks())){

	  for(TrackCollection::const_iterator trkcoll_ite=refittedTracks.begin(); trkcoll_ite!=refittedTracks.end(); trkcoll_ite++){
	
	    const TrackBaseRef retrackbaseref = displVtx.originalTrack(*trkcoll_ite); 

	    if(displVtx.isIncomingTrack(retrackbaseref)){

              VertexTrackQuality VOAssociation = PF_PU_AssoMapAlgos::TrackWeightAssociation(retrackbaseref.castTo<TrackRef>(), vtxcollH);

	      if(VOAssociation.second.second == 0.){
                pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *VOAssociation.first);
                return make_pair(VOAssociation.first, make_pair(trackref, -1.*IpPair.second.value()));
	      }

    	      TransientTrack transIncom(*retrackbaseref, &(*bFieldH) );
    	      transIncom.setBeamSpot(*bsH);
    	      transIncom.setES(iSetup);

	      VertexRef foundVertexRef = FindClosest3D(transIncom, vtxcollH, tWeight); 

              pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *foundVertexRef);	

	      return make_pair(foundVertexRef, make_pair(trackref, -1.*IpPair.second.value()));

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

        pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *foundVertexRef);	

	return make_pair(foundVertexRef, make_pair(trackref, -1.*IpPair.second.value()));

}


/*************************************************************************************/
/* function to check if a candidate is compatible with the BeamSpot                  */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::CheckBeamSpotCompability(TransientTrack transtrk, double cut)
{

        double relative_distance = transtrk.stateAtBeamLine().transverseImpactParameter().significance();

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
