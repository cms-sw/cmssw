
#include "CommonTools/RecoUtils/interface/PF_PU_AssoMapAlgos.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

using namespace edm;
using namespace std;
using namespace reco;

/*************************************************************************************/
/* dedicated constructor for the algorithms                                          */
/*************************************************************************************/

PF_PU_AssoMapAlgos::PF_PU_AssoMapAlgos(const edm::ParameterSet& iConfig, edm::ConsumesCollector & iC)
  : maxNumWarnings_(3),
    numWarnings_(0)
{

  	input_MaxNumAssociations_ = iConfig.getParameter<int>("MaxNumberOfAssociations");

  	token_VertexCollection_= iC.consumes<VertexCollection>(iConfig.getParameter<InputTag>("VertexCollection"));

  	token_BeamSpot_= iC.consumes<BeamSpot>(iConfig.getParameter<InputTag>("BeamSpot"));

  	input_doReassociation_= iConfig.getParameter<bool>("doReassociation");
  	cleanedColls_ = iConfig.getParameter<bool>("GetCleanedCollections");

  	ConversionsCollectionToken_= iC.consumes<ConversionCollection>(iConfig.getParameter<InputTag>("ConversionsCollection"));

  	KshortCollectionToken_= iC.consumes<VertexCompositeCandidateCollection>(iConfig.getParameter<InputTag>("V0KshortCollection"));
  	LambdaCollectionToken_= iC.consumes<VertexCompositeCandidateCollection>(iConfig.getParameter<InputTag>("V0LambdaCollection"));

  	NIVertexCollectionToken_= iC.consumes<PFDisplacedVertexCollection>(iConfig.getParameter<InputTag>("NIVertexCollection"));

  	input_FinalAssociation_= iConfig.getUntrackedParameter<int>("FinalAssociation", 0);

  	ignoremissingpfcollection_ = iConfig.getParameter<bool>("ignoreMissingCollection");

  	input_nTrack_ = iConfig.getParameter<double>("nTrackWeight");

}

/*************************************************************************************/
/* get all needed collections at the beginning                                       */
/*************************************************************************************/

void
PF_PU_AssoMapAlgos::GetInputCollections(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  	//get the offline beam spot
  	iEvent.getByToken(token_BeamSpot_, beamspotH);

  	//get the conversion collection for the gamma conversions
  	iEvent.getByToken(ConversionsCollectionToken_, convCollH);
	cleanedConvCollP = PF_PU_AssoMapAlgos::GetCleanedConversions(convCollH,beamspotH,cleanedColls_);

  	//get the vertex composite candidate collection for the Kshort's
  	iEvent.getByToken(KshortCollectionToken_, vertCompCandCollKshortH);
	cleanedKshortCollP = PF_PU_AssoMapAlgos::GetCleanedKshort(vertCompCandCollKshortH,beamspotH,cleanedColls_);

  	//get the vertex composite candidate collection for the Lambda's
  	iEvent.getByToken(LambdaCollectionToken_, vertCompCandCollLambdaH);
	cleanedLambdaCollP = PF_PU_AssoMapAlgos::GetCleanedLambda(vertCompCandCollLambdaH,beamspotH,cleanedColls_);

  	//get the displaced vertex collection for nuclear interactions
  	//create a new bool, true if no displaced vertex collection is in the event, mostly for AOD
  	missingColls = false;
  	if(!iEvent.getByToken(NIVertexCollectionToken_,displVertexCollH)){
          if (ignoremissingpfcollection_){

    	    missingColls = true;

            if ( numWarnings_ < maxNumWarnings_ ) {
	      LogWarning("PF_PU_AssoMapAlgos::GetInputCollections")
	        << "No Extra objects available in input file --> skipping reconstruction of displaced vertices !!" << endl;
	      ++numWarnings_;
            }

  	  }
	} else {

	    cleanedNICollP = PF_PU_AssoMapAlgos::GetCleanedNI(displVertexCollH,beamspotH,true);

	}

  	//get the input vertex collection
  	iEvent.getByToken(token_VertexCollection_, vtxcollH);

     	iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);

}

/*************************************************************************************/
/* create the track to vertex association map                                        */
/*************************************************************************************/
std::auto_ptr<TrackToVertexAssMap>
PF_PU_AssoMapAlgos::CreateTrackToVertexMap(edm::Handle<reco::TrackCollection> trkcollH, const edm::EventSetup& iSetup)
{

	auto_ptr<TrackToVertexAssMap> track2vertex(new TrackToVertexAssMap(vtxcollH, trkcollH));

	int num_vertices = vtxcollH->size();
	if ( num_vertices < input_MaxNumAssociations_) input_MaxNumAssociations_ = num_vertices;

  	//loop over all tracks of the track collection
  	for ( size_t idxTrack = 0; idxTrack < trkcollH->size(); ++idxTrack ) {

    	  TrackRef trackref = TrackRef(trkcollH, idxTrack);

          TransientTrack transtrk(trackref, &(*bFieldH) );
          transtrk.setBeamSpot(*beamspotH);
          transtrk.setES(iSetup);

	  vector<VertexRef>* vtxColl_help = CreateVertexVector(vtxcollH);

	  for ( int assoc_ite = 0; assoc_ite < input_MaxNumAssociations_; ++assoc_ite ) {

    	    VertexStepPair assocVtx = FindAssociation(trackref, vtxColl_help, bFieldH, iSetup, beamspotH, assoc_ite);
	    int step = assocVtx.second;
	    double distance = ( IPTools::absoluteImpactParameter3D( transtrk, *(assocVtx.first) ) ).second.value();

	    int quality = DefineQuality(assoc_ite, step, distance);

    	    //std::cout << "associating track: Pt = " << trackref->pt() << ","
    	    //	        << " eta = " << trackref->eta() << ", phi = " << trackref->phi()
    	    //	        << " to vertex: z = " << associatedVertex.first->position().z() << " with quality q = " << quality << std::endl;


    	    // Insert the best vertex and the pair of track and the quality of this association in the map
    	    track2vertex->insert( assocVtx.first, make_pair(trackref, quality) );

	    PF_PU_AssoMapAlgos::EraseVertex(vtxColl_help, assocVtx.first);

	  }

	  delete vtxColl_help;

  	}

	return track2vertex;

}

/*************************************************************************************/
/* create the vertex to track association map                                        */
/*************************************************************************************/

std::auto_ptr<VertexToTrackAssMap>
PF_PU_AssoMapAlgos::CreateVertexToTrackMap(edm::Handle<reco::TrackCollection> trkcollH, const edm::EventSetup& iSetup)
{

  	auto_ptr<VertexToTrackAssMap> vertex2track(new VertexToTrackAssMap(trkcollH, vtxcollH));

	int num_vertices = vtxcollH->size();
	if ( num_vertices < input_MaxNumAssociations_) input_MaxNumAssociations_ = num_vertices;

  	//loop over all tracks of the track collection
  	for ( size_t idxTrack = 0; idxTrack < trkcollH->size(); ++idxTrack ) {

    	  TrackRef trackref = TrackRef(trkcollH, idxTrack);

          TransientTrack transtrk(trackref, &(*bFieldH) );
          transtrk.setBeamSpot(*beamspotH);
          transtrk.setES(iSetup);

	  vector<VertexRef>* vtxColl_help = CreateVertexVector(vtxcollH);

	  for ( int assoc_ite = 0; assoc_ite < input_MaxNumAssociations_; ++assoc_ite ) {

    	    VertexStepPair assocVtx = FindAssociation(trackref, vtxColl_help, bFieldH, iSetup, beamspotH, assoc_ite);
	    int step = assocVtx.second;
	    double distance = ( IPTools::absoluteImpactParameter3D( transtrk, *(assocVtx.first) ) ).second.value();

	    int quality = DefineQuality(assoc_ite, step, distance);

    	    //std::cout << "associating track: Pt = " << trackref->pt() << ","
    	    //	        << " eta = " << trackref->eta() << ", phi = " << trackref->phi()
    	    //	        << " to vertex: z = " << associatedVertex.first->position().z() << " with quality q = " << quality << std::endl;

    	    // Insert the best vertex and the pair of track and the quality of this association in the map
    	    vertex2track->insert( trackref, make_pair(assocVtx.first, quality) );

	    PF_PU_AssoMapAlgos::EraseVertex(vtxColl_help, assocVtx.first);

	  }

	  delete vtxColl_help;

	}

	return vertex2track;

}

/*****************************************************************************************/
/* function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2  */
/*****************************************************************************************/

unique_ptr<TrackToVertexAssMap>
PF_PU_AssoMapAlgos::SortAssociationMap(TrackToVertexAssMap* trackvertexassInput, edm::Handle<reco::TrackCollection> trkcollH)
{
	//create a new TrackVertexAssMap for the Output which will be sorted
	unique_ptr<TrackToVertexAssMap> trackvertexassOutput(new TrackToVertexAssMap(vtxcollH, trkcollH));

	//Create and fill a vector of pairs of vertex and the summed (pT-pT_Error)**2 of the tracks associated to the vertex
	VertexPtsumVector vertexptsumvector;

	//loop over all vertices in the association map
        for(TrackToVertexAssMap::const_iterator assomap_ite=trackvertexassInput->begin(); assomap_ite!=trackvertexassInput->end(); assomap_ite++){

	  const VertexRef assomap_vertexref = assomap_ite->key;
  	  const TrackQualityPairVector trckcoll = assomap_ite->val;

	  float ptsum = 0;

	  TrackRef trackref;

	  //get the tracks associated to the vertex and calculate the manipulated pT**2
	  for(unsigned int trckcoll_ite=0; trckcoll_ite<trckcoll.size(); trckcoll_ite++){

	    trackref = trckcoll[trckcoll_ite].first;
	    int quality = trckcoll[trckcoll_ite].second;

	    if ( quality<=2 ) continue;

	    double man_pT = trackref->pt() - trackref->ptError();
	    if(man_pT>0.) ptsum+=man_pT*man_pT;

	  }

	  vertexptsumvector.push_back(make_pair(assomap_vertexref,ptsum));

	}

	while (!vertexptsumvector.empty()){

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
          for(TrackToVertexAssMap::const_iterator assomap_ite=trackvertexassInput->begin(); assomap_ite!=trackvertexassInput->end(); assomap_ite++){

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

/********************/
/*                  */
/* Member Functions */
/*                  */
/********************/

/*************************************************************************************/
/* create helping vertex vector to remove associated vertices                        */
/*************************************************************************************/

std::vector<reco::VertexRef>*
PF_PU_AssoMapAlgos::CreateVertexVector(edm::Handle<reco::VertexCollection> vtxcollH)
{

	vector<VertexRef>* output = new vector<VertexRef>();

  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);

	  output->push_back(vertexref);

	}

	return output;

}

/****************************************************************************/
/* erase one vertex from the vertex vector                                  */
/****************************************************************************/

void
PF_PU_AssoMapAlgos::EraseVertex(std::vector<reco::VertexRef>* vtxcollV, reco::VertexRef toErase)
{

  	for(unsigned int index_vtx=0;  index_vtx<vtxcollV->size(); ++index_vtx){

          VertexRef vertexref = vtxcollV->at(index_vtx);

	  if ( vertexref == toErase ){
            vtxcollV->erase(vtxcollV->begin()+index_vtx);
	    break;
	  }

	}

}


/*************************************************************************************/
/* function to find the closest vertex in 3D for a certain track                     */
/*************************************************************************************/

VertexRef
PF_PU_AssoMapAlgos::FindClosestZ(const reco::TrackRef trkref, std::vector<reco::VertexRef>* vtxcollV, double tWeight)
{

	double ztrack = trkref->vertex().z();

	VertexRef foundVertexRef = vtxcollV->at(0);

	double dzmin = 1e5;

	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollV->size(); ++index_vtx){

          VertexRef vertexref = vtxcollV->at(index_vtx);

	  double nTracks = sqrt(vertexref->tracksSize());

          double z_distance = fabs(ztrack - vertexref->z());

	  double weightedDistance = z_distance-tWeight*nTracks;

          if(weightedDistance<dzmin) {
            dzmin = weightedDistance;
            foundVertexRef = vertexref;
          }

	}

	return foundVertexRef;
}


/*************************************************************************************/
/* function to find the closest vertex in 3D for a certain track                     */
/*************************************************************************************/

VertexRef
PF_PU_AssoMapAlgos::FindClosest3D(TransientTrack transtrk, std::vector<reco::VertexRef>* vtxcollV, double tWeight)
{

	VertexRef foundVertexRef = vtxcollV->at(0);

	double d3min = 1e5;

	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollV->size(); ++index_vtx){

          VertexRef vertexref = vtxcollV->at(index_vtx);

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

/****************************************************************************************/
/* function to calculate the deltaR between a vector and a vector connecting two points */
/****************************************************************************************/

double
PF_PU_AssoMapAlgos::dR(const math::XYZPoint& vtx_pos, const math::XYZVector& vtx_mom, edm::Handle<reco::BeamSpot> bsH)
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
PF_PU_AssoMapAlgos::ComesFromConversion(const TrackRef trackref, const ConversionCollection& cleanedConvColl, Conversion* gamma)
{

	for(unsigned int convcoll_ite=0; convcoll_ite<cleanedConvColl.size(); convcoll_ite++){

	  if(ConversionTools::matchesConversion(trackref,cleanedConvColl.at(convcoll_ite))){

	    *gamma = cleanedConvColl.at(convcoll_ite);
	    return true;

  	  }

  	}

	return false;
}


/********************************************************************************/
/* function to find the closest vertex for a track from a conversion            */
/********************************************************************************/

VertexRef
PF_PU_AssoMapAlgos::FindConversionVertex(const reco::TrackRef trackref, const reco::Conversion& gamma, ESHandle<MagneticField> bfH, const EventSetup& iSetup, edm::Handle<reco::BeamSpot> bsH, std::vector<reco::VertexRef>* vtxcollV, double tWeight)
{

	math::XYZPoint conv_pos = gamma.conversionVertex().position();

	math::XYZVector conv_mom(gamma.refittedPair4Momentum().x(),
	                         gamma.refittedPair4Momentum().y(),
	                         gamma.refittedPair4Momentum().z());

	Track photon(trackref->chi2(), trackref->ndof(), conv_pos, conv_mom, 0, trackref->covariance());

    	TransientTrack transpho(photon, &(*bfH) );
    	transpho.setBeamSpot(*bsH);
    	transpho.setES(iSetup);

	return FindClosest3D(transpho, vtxcollV, tWeight);

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
PF_PU_AssoMapAlgos::ComesFromV0Decay(const TrackRef trackref, const VertexCompositeCandidateCollection& cleanedKshort, const VertexCompositeCandidateCollection& cleanedLambda, VertexCompositeCandidate* V0)
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

VertexRef
PF_PU_AssoMapAlgos::FindV0Vertex(const TrackRef trackref, const VertexCompositeCandidate& V0_vtx, ESHandle<MagneticField> bFieldH, const EventSetup& iSetup, Handle<BeamSpot> bsH, std::vector<reco::VertexRef>* vtxcollV, double tWeight)
{

	const math::XYZPoint& dec_pos = V0_vtx.vertex();

	math::XYZVector dec_mom(V0_vtx.momentum().x(),
	                        V0_vtx.momentum().y(),
	                        V0_vtx.momentum().z());

	Track V0(trackref->chi2(), trackref->ndof(), dec_pos, dec_mom, 0, trackref->covariance());

    	TransientTrack transV0(V0, &(*bFieldH) );
    	transV0.setBeamSpot(*bsH);
    	transV0.setES(iSetup);

	return FindClosest3D(transV0, vtxcollV, tWeight);

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
PF_PU_AssoMapAlgos::ComesFromNI(const TrackRef trackref, const PFDisplacedVertexCollection& cleanedNI, PFDisplacedVertex* displVtx)
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

VertexRef
PF_PU_AssoMapAlgos::FindNIVertex(const TrackRef trackref, const PFDisplacedVertex& displVtx, ESHandle<MagneticField> bFieldH, const EventSetup& iSetup, Handle<BeamSpot> bsH, std::vector<reco::VertexRef>* vtxcollV, double tWeight)
{

	TrackCollection refittedTracks = displVtx.refittedTracks();

	if((displVtx.isTherePrimaryTracks()) || (displVtx.isThereMergedTracks())){

	  for(TrackCollection::const_iterator trkcoll_ite=refittedTracks.begin(); trkcoll_ite!=refittedTracks.end(); trkcoll_ite++){

	    const TrackBaseRef retrackbaseref = displVtx.originalTrack(*trkcoll_ite);

	    if(displVtx.isIncomingTrack(retrackbaseref)){

              VertexRef VOAssociation = TrackWeightAssociation(retrackbaseref, vtxcollV);

	      if( VOAssociation->trackWeight(retrackbaseref) >= 1.e-5 ){
	        return VOAssociation;
	      }

    	      TransientTrack transIncom(*retrackbaseref, &(*bFieldH) );
    	      transIncom.setBeamSpot(*bsH);
    	      transIncom.setES(iSetup);

	      return FindClosest3D(transIncom, vtxcollV, tWeight);

	    }

	  }

	}

	const math::XYZPoint& ni_pos = displVtx.position();

	math::XYZVector ni_mom(displVtx.primaryMomentum().x(),
	                       displVtx.primaryMomentum().y(),
	                       displVtx.primaryMomentum().z());

	Track incom(trackref->chi2(), trackref->ndof(), ni_pos, ni_mom, 0, trackref->covariance());

    	TransientTrack transIncom(incom, &(*bFieldH) );
    	transIncom.setBeamSpot(*bsH);
    	transIncom.setES(iSetup);

	return FindClosest3D(transIncom, vtxcollV, tWeight);

}

/*************************************************************************************/
/* function to find the vertex with the highest TrackWeight for a certain track      */
/*************************************************************************************/

VertexRef
PF_PU_AssoMapAlgos::TrackWeightAssociation(const TrackBaseRef& trackbaseRef, std::vector<reco::VertexRef>* vtxcollV)
{

	VertexRef bestvertexref = vtxcollV->at(0);
 	float bestweight = 0.;

	//loop over all vertices in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollV->size(); ++index_vtx){

          VertexRef vertexref = vtxcollV->at(index_vtx);

     	  //get the most probable vertex for the track
	  float weight = vertexref->trackWeight(trackbaseRef);
	  if(weight>bestweight){
  	    bestweight = weight;
	    bestvertexref = vertexref;
 	  }

	}

  	return bestvertexref;

}

/*************************************************************************************/
/* find an association for a certain track                                           */
/*************************************************************************************/

VertexStepPair
PF_PU_AssoMapAlgos::FindAssociation(const reco::TrackRef& trackref, std::vector<reco::VertexRef>* vtxColl, edm::ESHandle<MagneticField> bfH, const edm::EventSetup& iSetup, edm::Handle<reco::BeamSpot> bsH, int assocNum)
{

	const TrackBaseRef& trackbaseRef = TrackBaseRef(trackref);

	VertexRef foundVertex;

	//if it is not the first try of an association jump to the final association
	//to avoid multiple (secondary) associations and/or unphysical (primary and secondary) associations
	if ( assocNum>0 ) goto finalStep;

	// Step 1: First round of association:
    	// Find the vertex with the highest track-to-vertex association weight
    	foundVertex = TrackWeightAssociation(trackbaseRef, vtxColl);

    	if ( foundVertex->trackWeight(trackbaseRef) >= 1.e-5 ){
          return make_pair( foundVertex, 0. );
	}

	// Step 2: Reassociation
    	// Second round of association:
    	// In case no vertex with track-to-vertex association weight > 1.e-5 is found,
    	// check the track originates from a neutral hadron decay, photon conversion or nuclear interaction

	if ( input_doReassociation_ ) {

      	  // Test if the track comes from a photon conversion:
      	  // If so, try to find the vertex of the mother particle
	  Conversion gamma;
          if ( ComesFromConversion(trackref, *cleanedConvCollP, &gamma) ){
  	    foundVertex = FindConversionVertex(trackref, gamma, bfH, iSetup, bsH, vtxColl, input_nTrack_);
            return make_pair( foundVertex, 1. );
          }

      	  // Test if the track comes from a Kshort or Lambda decay:
      	  // If so, reassociate the track to the vertex of the V0
	  VertexCompositeCandidate V0;
	  if ( ComesFromV0Decay(trackref, *cleanedKshortCollP, *cleanedLambdaCollP, &V0) ) {
            foundVertex = FindV0Vertex(trackref, V0, bfH, iSetup, bsH, vtxColl, input_nTrack_);
            return make_pair( foundVertex, 1. );
	  }

	  if ( !missingColls ) {

      	    // Test if the track comes from a nuclear interaction:
      	    // If so, reassociate the track to the vertex of the incoming particle
	    PFDisplacedVertex displVtx;
	    if ( ComesFromNI(trackref, *cleanedNICollP, &displVtx) ){
 	      foundVertex = FindNIVertex(trackref, displVtx, bfH, iSetup, bsH, vtxColl, input_nTrack_);
              return make_pair( foundVertex, 1. );
	    }

	  }

	}

	// Step 3: Final association
      	// If no vertex is found with track-to-vertex association weight > 1.e-5
      	// and no reassociation was done do the final association
	// look for the closest vertex in 3D or in z/longitudinal distance
	// or associate the track always to the first vertex (default)

	finalStep:

	switch (input_FinalAssociation_) {

 	  case 1:{

	    // closest in z
	    foundVertex = FindClosestZ(trackref,vtxColl,input_nTrack_);
	    break;


          }

 	  case 2:{

	    // closest in 3D
            TransientTrack transtrk(trackref, &(*bfH) );
            transtrk.setBeamSpot(*bsH);
            transtrk.setES(iSetup);

	    foundVertex = FindClosest3D(transtrk,vtxColl,input_nTrack_);
	    break;

          }

 	  default:{

	    // allways first vertex
            foundVertex = vtxColl->at(0);
	    break;

          }

	}

	return make_pair( foundVertex, 2. );

}

/*************************************************************************************/
/* get the quality for a certain association                                         */
/*************************************************************************************/

int
PF_PU_AssoMapAlgos::DefineQuality(int assoc_ite, int step, double distance)
{

	int quality = 0;

	switch (step) {

	  case 0:{

	    //TrackWeight association
            if ( distance <= tw_90 ) {
              quality = 5;
	    } else {
	      if ( distance <= tw_70 ) {
                quality = 4;
	      } else {
	        if ( distance <= tw_50 ) {
                  quality = 3;
	        } else {
                  quality = 2;
	        }
	      }
	    }
	    break;

	  }

	  case 1:{

      	    //Secondary association
            if ( distance <= sec_70 ) {
              quality = 4;
	    } else {
	      if ( distance <= sec_50 ) {
                quality = 3;
	      } else {
                quality = 2;
	      }
	    }
	    break;

	  }

	  case 2:{

	    //Final association
            if ( assoc_ite == 1 ) {
              quality = 1;
	    } else {
              if ( assoc_ite >= 2 ) {
                quality = 0;
	      } else {
                if ( distance <= fin_70 ) {
                  quality = 4;
	        } else {
	          if ( distance <= fin_50 ) {
                    quality = 3;
	          } else {
                    quality = 2;
	          }
	        }
	      }
	    }
	    break;

	  }

	  default:{

            quality = -1;
	    break;
     	  }

	}

	return quality;

}
