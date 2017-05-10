
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

PF_PU_AssoMapAlgos::PF_PU_AssoMapAlgos(const edm::ParameterSet& iConfig)
  : maxNumWarnings_(3),
    numWarnings_(0)
{

  input_MaxNumAssociations_ = iConfig.getParameter<int>("MaxNumberOfAssociations");

  input_VertexCollection_= iConfig.getParameter<InputTag>("VertexCollection");

  input_BeamSpot_= iConfig.getParameter<InputTag>("BeamSpot");

  input_doReassociation_= iConfig.getParameter<bool>("doReassociation");
  cleanedColls_ = iConfig.getParameter<bool>("GetCleanedCollections");

  ConversionsCollection_= iConfig.getParameter<InputTag>("ConversionsCollection");

  KshortCollection_= iConfig.getParameter<InputTag>("V0KshortCollection");
  LambdaCollection_= iConfig.getParameter<InputTag>("V0LambdaCollection");

  NIVertexCollection_= iConfig.getParameter<InputTag>("NIVertexCollection");

  IFVVertexCollection_ = iConfig.getParameter<InputTag>("IVFVertexCollection");
  if ( IFVVertexCollection_.label()=="" ) {
    LogWarning("PF_PU_AssoMapAlgos::PF_PU_AssoMapAlgos")  << "No InputTag for IV's given --> skipping reconstruction of inclusive vertices !!" << endl;
  }

  input_FinalAssociation_= iConfig.getUntrackedParameter<int>("FinalAssociation", 0);

  ignoremissingpfcollection_ = iConfig.getParameter<bool>("ignoreMissingCollection");

  input_nTrack_z_ = iConfig.getParameter<double>("nTrackWeight_z");
  input_nTrack_3D_ = iConfig.getParameter<double>("nTrackWeight_3D");

  missingColls = false;

}

/*************************************************************************************/
/* dedicated destructor for the algorithms                                          */
/*************************************************************************************/

PF_PU_AssoMapAlgos::~PF_PU_AssoMapAlgos()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

/*************************************************************************************/
/* get all needed collections at the beginning                                       */ 
/*************************************************************************************/

void 
PF_PU_AssoMapAlgos::GetInputCollections(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  //get the offline beam spot
  iEvent.getByLabel(input_BeamSpot_, beamspotH);

  GlobalPoint bsPosition = RecoVertex::convertPos( beamspotH->position() );
  GlobalError bsError = RecoVertex::convertError( beamspotH->covariance3D() );

  BSVertexState = VertexState( bsPosition, bsError );

  //get the conversion collection for the gamma conversions
  iEvent.getByLabel(ConversionsCollection_, convCollH);
  cleanedConvCollP = PF_PU_AssoMapAlgos::GetCleanedConversions(convCollH,cleanedColls_);

  //get the vertex composite candidate collection for the Kshort's
  iEvent.getByLabel(KshortCollection_, vertCompCandCollKshortH);
  cleanedKshortCollP = PF_PU_AssoMapAlgos::GetCleanedKshort(vertCompCandCollKshortH,cleanedColls_);

  //get the vertex composite candidate collection for the Lambda's
  iEvent.getByLabel(LambdaCollection_, vertCompCandCollLambdaH);
  cleanedLambdaCollP = PF_PU_AssoMapAlgos::GetCleanedLambda(vertCompCandCollLambdaH,cleanedColls_);

  //get the displaced vertex collection for nuclear interactions
  if ( !iEvent.getByLabel(NIVertexCollection_,displVertexCollH) ) {
    if ( ignoremissingpfcollection_ ){
      if ( numWarnings_ < maxNumWarnings_ ) {
        LogWarning("PF_PU_AssoMapAlgos::GetInputCollections")  << "No Extra objects available in input file --> skipping reconstruction of displaced vertices !!" << endl;
        ++numWarnings_;
      }
    }
  } else {

    cleanedNICollP = PF_PU_AssoMapAlgos::GetCleanedNI(displVertexCollH, true);

  }

  //get the inclusive vertex finder collection 
  if ( !iEvent.getByLabel(IFVVertexCollection_,ivfVertexCollH) ) {
    if ( ( ignoremissingpfcollection_ ) && !( IFVVertexCollection_.label()=="" ) ){
      if ( numWarnings_ < maxNumWarnings_ ) {
        LogWarning("PF_PU_AssoMapAlgos::GetInputCollections")  << "No Extra objects available in input file --> skipping reconstruction of ifv vertices !!" << endl;
        ++numWarnings_;
      }
    }
  } else {

    cleanedIVFCollP = PF_PU_AssoMapAlgos::GetCleanedIVF(ivfVertexCollH, true);

  }
	
  //get the input vertex collection
  iEvent.getByLabel(input_VertexCollection_, vtxcollH);

  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);

}

/*************************************************************************************/
/* create the track to vertex association map                                        */ 
/*************************************************************************************/
std::auto_ptr<TrackToVertexAssMap> 
PF_PU_AssoMapAlgos::CreateTrackToVertexMap(edm::Handle<reco::TrackCollection> trkcollH, const edm::EventSetup& iSetup)
{

  auto_ptr<TrackToVertexAssMap> track2vertex(new TrackToVertexAssMap());

  int num_associations = input_MaxNumAssociations_;
  int num_vertices = vtxcollH->size();
  if ( num_vertices < num_associations) num_associations = num_vertices;
	
  //loop over all tracks of the track collection	
  for ( size_t idxTrack = 0; idxTrack < trkcollH->size(); ++idxTrack ) {

    TrackRef trackref = TrackRef(trkcollH, idxTrack);

    TransientTrack transtrk(trackref, &(*bFieldH) );
    transtrk.setBeamSpot(*beamspotH);
    transtrk.setES(iSetup);

    vector<VertexRef>* vtxColl_help = CreateVertexVector(vtxcollH);
    StepDistancePairVector distances;

    int num_assoc = 0;

    for ( int assoc_ite = 0; assoc_ite < num_associations; ++assoc_ite ) {

      VertexStepPair assocVtx = FindAssociation(trackref, vtxColl_help, bFieldH, iSetup, beamspotH, assoc_ite);
      VertexRef associatedVertex = assocVtx.first;
      int step = assocVtx.second;
      double distance = ( IPTools::absoluteImpactParameter3D( transtrk, *associatedVertex ) ).second.value();
 
      int quality = DefineQuality(distances, step, distance);
      distances.push_back( make_pair(step, distance) );

      // edm::LogInfo("Trck2VtxAssociation") << "associating track: Pt = " << trackref->pt() << ","
      // << " eta = " << trackref->eta() << ", phi = " << trackref->phi()
      // << " to vertex: z = " << associatedVertex.first->position().z() << " with quality q = " << quality << std::endl;


      // Insert the best vertex and the pair of track and the quality of this association in the map
      track2vertex->insert( associatedVertex, make_pair(trackref, quality) );

      PF_PU_AssoMapAlgos::EraseVertex(vtxColl_help, associatedVertex);
      num_assoc++;

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

  auto_ptr<VertexToTrackAssMap> vertex2track(new VertexToTrackAssMap());

  int num_associations = input_MaxNumAssociations_;
  int num_vertices = vtxcollH->size();
  if ( num_vertices < num_associations) num_associations = num_vertices;
	
  //loop over all tracks of the track collection	
  for ( size_t idxTrack = 0; idxTrack < trkcollH->size(); ++idxTrack ) {

    TrackRef trackref = TrackRef(trkcollH, idxTrack);

    TransientTrack transtrk(trackref, &(*bFieldH) );
    transtrk.setBeamSpot(*beamspotH);
    transtrk.setES(iSetup);

    vector<VertexRef>* vtxColl_help = CreateVertexVector(vtxcollH);
    StepDistancePairVector distances;

    for ( int assoc_ite = 0; assoc_ite < num_associations; ++assoc_ite ) {

      VertexStepPair assocVtx = FindAssociation(trackref, vtxColl_help, bFieldH, iSetup, beamspotH, assoc_ite);	 
      VertexRef associatedVertex = assocVtx.first;
      int step = assocVtx.second;
      double distance = ( IPTools::absoluteImpactParameter3D( transtrk, *associatedVertex ) ).second.value();

      int quality = DefineQuality(distances, step, distance);
      distances.push_back( make_pair(step, distance) );

      // edm::LogInfo("Vtx2TrckAssociation") << "associating track: Pt = " << trackref->pt() << ","
      // << " eta = " << trackref->eta() << ", phi = " << trackref->phi()
      // << " to vertex: z = " << associatedVertex.first->position().z() << " with quality q = " << quality << std::endl;

      // Insert the best vertex and the pair of track and the quality of this association in the map
      vertex2track->insert( trackref, make_pair(associatedVertex, quality) );
 
      PF_PU_AssoMapAlgos::EraseVertex(vtxColl_help, associatedVertex);

    }

    delete vtxColl_help;

  }

  return vertex2track;

}

/*****************************************************************************************/
/* function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2  */ 
/*****************************************************************************************/

auto_ptr<TrackToVertexAssMap>  
PF_PU_AssoMapAlgos::SortAssociationMap(TrackToVertexAssMap* trackvertexassInput) 
{

  //create a new TrackVertexAssMap for the Output which will be sorted
  auto_ptr<TrackToVertexAssMap> trackvertexassOutput(new TrackToVertexAssMap() );

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

    double nTracks = sqrt( vertexref->tracksSize() );

    //find and store the closest vertex in z
    double distance = fabs(ztrack - vertexref->z());

    double weightedDistance = distance-tWeight*nTracks;	

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
  for ( unsigned int index_vtx=0;  index_vtx<vtxcollV->size(); ++index_vtx ) {

    VertexRef vertexref = vtxcollV->at( index_vtx );

    double nTracks = vertexref->tracksSize();

    double distance = 1e5;	        
    pair<bool,Measurement1D> IpPair = IPTools::absoluteImpactParameter3D(transtrk, *vertexref);

    if ( IpPair.first ) distance = IpPair.second.value();

    double weightedDistance = distance-tWeight*nTracks;	

    if ( weightedDistance<d3min ) {
      d3min = weightedDistance; 
      foundVertexRef = vertexref;
    }
 
  }

  return foundVertexRef;
  
}


/*************************************************************************************/
/* function to filter the conversion collection                                      */ 
/*************************************************************************************/

auto_ptr<ConversionCollection> 
PF_PU_AssoMapAlgos::GetCleanedConversions(edm::Handle<reco::ConversionCollection> convCollH, bool cleanedColl)
{

  auto_ptr<ConversionCollection> cleanedConvColl(new ConversionCollection() );

  for ( unsigned int convcoll_idx=0; convcoll_idx<convCollH->size(); convcoll_idx++ ){

    ConversionRef convref(convCollH,convcoll_idx);

    if ( !cleanedColl ) {   
      cleanedConvColl->push_back(*convref);
      continue;
    }

    if ( convref->quality( Conversion::highPurity ) ){

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

  math::XYZVector conv_mom( gamma.refittedPair4Momentum().x(),
   						    gamma.refittedPair4Momentum().y(),
   						    gamma.refittedPair4Momentum().z());

  Track photon( trackref->chi2(), trackref->ndof(), conv_pos, conv_mom, 0, trackref->covariance() );

  TransientTrack transpho(photon, &(*bfH) );
  transpho.setBeamSpot( *bsH );
  transpho.setES( iSetup );

  return FindClosest3D(transpho, vtxcollV, tWeight);	

}

/*************************************************************************************/
/* function to filter the Kshort collection                                          */ 
/*************************************************************************************/

auto_ptr<VertexCompositeCandidateCollection>
PF_PU_AssoMapAlgos::GetCleanedKshort(Handle<VertexCompositeCandidateCollection> KshortsH, bool cleanedColl)
{

  auto_ptr<VertexCompositeCandidateCollection> cleanedKaonColl(new VertexCompositeCandidateCollection() );

  for ( unsigned int kscoll_idx=0; kscoll_idx<KshortsH->size(); kscoll_idx++ ) {

    VertexCompositeCandidateRef ksref(KshortsH,kscoll_idx);

    if ( !cleanedColl ) {   
      cleanedKaonColl->push_back(*ksref);
      continue;
    }

    GlobalPoint dec_pos = RecoVertex::convertPos(ksref->vertex());    

    GlobalError decayVertexError = GlobalError(ksref->vertexCovariance(0,0), ksref->vertexCovariance(0,1), ksref->vertexCovariance(1,1), ksref->vertexCovariance(0,2), ksref->vertexCovariance(1,2), ksref->vertexCovariance(2,2));  

    double kaon_significance = ( distanceComputerXY.distance( BSVertexState, VertexState( dec_pos, decayVertexError ) ) ).significance();

    if ( ( ksref->vertexNormalizedChi2()<=7. ) &&
         ( fabs(ksref->mass() - kMass)<=0.06 ) &&
         ( kaon_significance>25. ) ) {

      cleanedKaonColl->push_back(*ksref);

    }

  }

  return cleanedKaonColl;
  
}

/*************************************************************************************/
/* function to filter the Lambda collection                                          */ 
/*************************************************************************************/

auto_ptr<VertexCompositeCandidateCollection>
PF_PU_AssoMapAlgos::GetCleanedLambda(Handle<VertexCompositeCandidateCollection> LambdasH, bool cleanedColl)
{

  auto_ptr<VertexCompositeCandidateCollection> cleanedLambdaColl(new VertexCompositeCandidateCollection() );

  for ( unsigned int lambdacoll_idx=0; lambdacoll_idx<LambdasH->size(); lambdacoll_idx++ ) {

    VertexCompositeCandidateRef lambdaref(LambdasH,lambdacoll_idx);

    if ( !cleanedColl ) {   
      cleanedLambdaColl->push_back(*lambdaref);
      continue;
    }

    GlobalPoint dec_pos = RecoVertex::convertPos(lambdaref->vertex());    

    GlobalError decayVertexError = GlobalError(lambdaref->vertexCovariance(0,0), lambdaref->vertexCovariance(0,1), lambdaref->vertexCovariance(1,1), lambdaref->vertexCovariance(0,2), lambdaref->vertexCovariance(1,2), lambdaref->vertexCovariance(2,2));  

    double lambda_significance = ( distanceComputerXY.distance( BSVertexState, VertexState( dec_pos, decayVertexError ) ) ).significance();

    if ( ( lambdaref->vertexNormalizedChi2()<=7. ) &&
         ( fabs(lambdaref->mass() - lamMass)<=0.04 ) &&
         ( lambda_significance>26. ) ){

      cleanedLambdaColl->push_back(*lambdaref);

    }

  }

  return cleanedLambdaColl;
  
}

/*************************************************************************************/
/* function to find out if the track comes from a V0 decay                           */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::ComesFromV0Decay(const TrackRef trackref, const VertexCompositeCandidateCollection& cleanedVCCC, VertexCompositeCandidate* V0)
{

  //the part for the reassociation of particles from V= decays
  for(VertexCompositeCandidateCollection::const_iterator iV0=cleanedVCCC.begin(); iV0!=cleanedVCCC.end(); iV0++){

    const RecoChargedCandidate *dauCand1 = dynamic_cast<const RecoChargedCandidate*>(iV0->daughter(0));
    TrackRef dauTk1 = dauCand1->track();
    const RecoChargedCandidate *dauCand2 = dynamic_cast<const RecoChargedCandidate*>(iV0->daughter(1));
    TrackRef dauTk2 = dauCand2->track();

    if ( (trackref==dauTk1) || (trackref==dauTk2) ) {

      *V0 = *iV0; 
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

  math::XYZPoint dec_pos = V0_vtx.vertex();

  math::XYZVector dec_mom( V0_vtx.momentum().x(),
						   V0_vtx.momentum().y(),
						   V0_vtx.momentum().z() );

  Track V0(trackref->chi2(), trackref->ndof(), dec_pos, dec_mom, 0, trackref->covariance());
 
  TransientTrack transV0(V0, &(*bFieldH) );
  transV0.setBeamSpot( *bsH );
  transV0.setES( iSetup );

  return FindClosest3D(transV0, vtxcollV, tWeight);		

}


/*************************************************************************************/
/* function to filter the nuclear interaction collection                             */ 
/*************************************************************************************/

auto_ptr<PFDisplacedVertexCollection>
PF_PU_AssoMapAlgos::GetCleanedNI(Handle<PFDisplacedVertexCollection> NuclIntH, bool cleanedColl)
{

  auto_ptr<PFDisplacedVertexCollection> cleanedNIColl(new PFDisplacedVertexCollection() );

  for ( PFDisplacedVertexCollection::const_iterator niref=NuclIntH->begin(); niref!=NuclIntH->end(); niref++ ) {

    if ( !cleanedColl ) {
      cleanedNIColl->push_back(*niref);
      continue;
    }

    GlobalPoint ni_pos = RecoVertex::convertPos( niref->position() );    
    GlobalError interactionVertexError = RecoVertex::convertError( niref->error() );

    double nuclint_distance = ( distanceComputerXY.distance( BSVertexState, VertexState( ni_pos, interactionVertexError ) ) ).value();

    if ( ( !niref->isFake() ) &&
         ( niref->isNucl() ) &&
         ( niref->normalizedChi2()<=2. ) &&
         ( niref->tracksSize()>=2 ) &&
         ( nuclint_distance>3. ) ) {

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
  for ( PFDisplacedVertexCollection::const_iterator iDisplV=cleanedNI.begin(); iDisplV!=cleanedNI.end(); iDisplV++ ) {

    if ( iDisplV->trackWeight(trackref)>1.e-5 ) {

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
PF_PU_AssoMapAlgos::FindNIVertex(const TrackRef trackref, const PFDisplacedVertex& displVtx, ESHandle<MagneticField> bFieldH, const EventSetup& iSetup, Handle<BeamSpot> bsH, std::vector<reco::VertexRef>* vtxcollV, double tWeight, TransientTrack transhelp)
{

  TrackCollection refittedTracks = displVtx.refittedTracks();

  if ( ( displVtx.isTherePrimaryTracks() ) || ( displVtx.isThereMergedTracks() ) ){

    for ( TrackCollection::const_iterator trkcoll_ite=refittedTracks.begin(); trkcoll_ite!=refittedTracks.end(); trkcoll_ite++ ) {

      const TrackBaseRef retrackbaseref = displVtx.originalTrack(*trkcoll_ite); 

      if ( displVtx.isIncomingTrack( retrackbaseref ) ) {

        VertexRef bestvertexref = vtxcollV->at(0);
        float bestweight = 0.;
 
        for ( unsigned int index_vtx=0; index_vtx<vtxcollV->size(); ++index_vtx ) {

          VertexRef vertexref = vtxcollV->at(index_vtx);

          //get the most probable vertex for the track
          float weight = vertexref->trackWeight(retrackbaseref);
          if(weight>bestweight){
            bestweight = weight;
            bestvertexref = vertexref;
          } 

        }

        if ( bestweight>1.e-5 ) return bestvertexref;
 
        TransientTrack transIncom(*retrackbaseref, &(*bFieldH) );
        transIncom.setBeamSpot( *bsH );
        transIncom.setES( iSetup );

        return FindClosest3D(transIncom, vtxcollV, tWeight); 

      }

    }

  }

  return FindClosest3D(transhelp, vtxcollV, tWeight);  

}


/*************************************************************************************/
/* function to filter the inclusive vertex finder collection                         */ 
/*************************************************************************************/

auto_ptr<VertexCollection>
PF_PU_AssoMapAlgos::GetCleanedIVF(Handle<VertexCollection> ifvH, bool cleanedColl)
{

  auto_ptr<VertexCollection> cleanedIVFColl(new VertexCollection() );
 
  for ( VertexCollection::const_iterator ivfref=ifvH->begin(); ivfref!=ifvH->end(); ivfref++ ) {

    if ( !cleanedColl ) {
      cleanedIVFColl->push_back(*ivfref);
      continue;
    } 

    GlobalPoint iv_pos = RecoVertex::convertPos( ivfref->position() );    
    GlobalError iv_err = RecoVertex::convertError( ivfref->error() );  
    
    double ivf_significance = ( distanceComputerXY.distance( BSVertexState, VertexState( iv_pos, iv_err ))).significance();
    
    if ( ( ivfref->isValid() ) && 
  	     ( !ivfref->isFake() ) && 
  	     ( ivfref->chi2()<=10. ) && 
  	     ( ivfref->nTracks(0.)>=2 ) && 
  	     ( ivf_significance>=5. ) ) {
  	     
      cleanedIVFColl->push_back(*ivfref);
 
    }            
 
  }

  return cleanedIVFColl;
  
}

/*************************************************************************************/
/* function to find out if the track comes from a inclusive vertex                   */ 
/*************************************************************************************/

bool
PF_PU_AssoMapAlgos::ComesFromIVF(const TrackRef trackref, const VertexCollection& cleanedIVF, Vertex* ivfVtx)
{

  for(VertexCollection::const_iterator iInclV=cleanedIVF.begin(); iInclV!=cleanedIVF.end(); iInclV++){

    if(iInclV->trackWeight(trackref)>1.e-5){

      *ivfVtx = *iInclV; 
      return true;

    }

  }

  return false;
  
}

/*************************************************************************************/
/* function to find the closest vertex in z for a track from an inclusive vertex     */ 
/*************************************************************************************/

VertexRef
PF_PU_AssoMapAlgos::FindIVFVertex(const TrackRef trackref, const Vertex& ivfVtx, ESHandle<MagneticField> bFieldH, const EventSetup& iSetup, Handle<BeamSpot> bsH, std::vector<reco::VertexRef>* vtxcollV, double tWeight)
{

  math::XYZPoint iv_pos = ivfVtx.position();

  math::XYZVector iv_mom( ivfVtx.p4(0.1, 0.).x(),
						  ivfVtx.p4(0.1, 0.).y(),
						  ivfVtx.p4(0.1, 0.).z() );
  
  Track incom(trackref->chi2(), trackref->ndof(), iv_pos, iv_mom, 0, trackref->covariance());

  TransientTrack transIncom(incom, &(*bFieldH) );
  transIncom.setBeamSpot( *bsH );
  transIncom.setES( iSetup );

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

  TransientTrack transtrk(trackref, &(*bfH) );
  transtrk.setBeamSpot(*bsH);
  transtrk.setES(iSetup);

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
    if ( PF_PU_AssoMapAlgos::ComesFromConversion(trackref, *cleanedConvCollP, &gamma) ){
      foundVertex = PF_PU_AssoMapAlgos::FindConversionVertex(trackref, gamma, bfH, iSetup, bsH, vtxColl, input_nTrack_3D_);
      return make_pair( foundVertex, 1. );
    }

    // Test if the track comes from a Kshort or Lambda decay:
    // If so, reassociate the track to the vertex of the V0
    VertexCompositeCandidate V0;
    if ( ( PF_PU_AssoMapAlgos::ComesFromV0Decay(trackref, *cleanedKshortCollP, &V0) ) ||
         ( PF_PU_AssoMapAlgos::ComesFromV0Decay(trackref, *cleanedLambdaCollP, &V0) ) ) {
      foundVertex = PF_PU_AssoMapAlgos::FindV0Vertex(trackref, V0, bfH, iSetup, bsH, vtxColl, input_nTrack_3D_);
      return make_pair( foundVertex, 1. );
    }

    if ( displVertexCollH.isValid() ) {

      // Test if the track comes from a nuclear interaction:
      // If so, reassociate the track to the vertex of the incoming particle 
      PFDisplacedVertex displVtx;
      if ( PF_PU_AssoMapAlgos::ComesFromNI(trackref, *cleanedNICollP, &displVtx) ){
        foundVertex = PF_PU_AssoMapAlgos::FindNIVertex(trackref, displVtx, bfH, iSetup, bsH, vtxColl, input_nTrack_3D_, transtrk);
        return make_pair( foundVertex, 1. );
      }

    }

    if ( ivfVertexCollH.isValid() ) {

      // Test if the track comes from a nuclear interaction:
      // If so, reassociate the track to the vertex of the incoming particle 
      Vertex ivfVtx;
      if ( ComesFromIVF(trackref, *cleanedIVFCollP, &ivfVtx) ){
        foundVertex = FindIVFVertex(trackref, ivfVtx, bfH, iSetup, bsH, vtxColl, input_nTrack_3D_);
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
        foundVertex = FindClosestZ(trackref, vtxColl, input_nTrack_z_);
        break;

      }

      case 2:{

        // closest in 3D
        TransientTrack transtrk(trackref, &(*bfH) );
        transtrk.setBeamSpot(*bsH);
        transtrk.setES(iSetup);

        foundVertex = FindClosest3D(transtrk, vtxColl, input_nTrack_3D_);
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
PF_PU_AssoMapAlgos::DefineQuality(vector< pair<int, double> > distances, int step, double distance)
{

  int quality = 0;
  int assoc_ite = distances.size();
	
  if ( assoc_ite >= 2 ) return 0;

  switch ( step ) {
	    
    case 0:{

      //TrackWeight association
      if ( distance <= tw_1st_90_cum ) {
        quality = 6;
      } else {
	quality = 5;
      }
      break;

    }

    case 1:{

      //Secondary association
      if ( distance <= sc_1st_70_cum ) {
        quality = 5;
      } else {
	quality = 4;
      }
      break;

    }

    case 2:{

      //Final association
      if ( input_FinalAssociation_ == 0 ) {

        // always first vertex
	if ( assoc_ite == 0 ) {

	  if ( distance <= f1_1st_70_cum ) {
	    quality = 5;
	  } else {
	    if ( distance <= f1_1st_50_cum ) {
	      quality = 4;
	    } else {
	      quality = 3;
	    }
	  }

	} else {

	  int firstStep = distances.at(0).first;
	  int firstDistance = distances.at(0).second;

	  switch ( firstStep ) {

	    case 0:{

	      if ( ( firstDistance <= tw_1st_50 ) &&
	           ( distance <= tw_2nd_f1_cum ) ) {
	        quality = 2;
	      } else {
	        quality = 1;
	      }
	      break;

	    }

	    case 1:{

              if ( ( ( firstDistance <= sc_1st_50 ) && ( distance <= sc_2nd_f1_0_cum  ) ) ||
                   ( ( firstDistance > sc_1st_50 )  && ( distance <= sc_2nd_f1_1_cum  ) ) ) {
                quality = 2;
              } else {
                quality = 1;
              }
              break;

	    }

	    case 2:{
              quality = 1;
	    }

	    default:{
	      quality = 1;
	    }

	  }

        }

      } else {

        if ( input_FinalAssociation_ == 1 ) {

	  // closest in z
	  if ( assoc_ite == 0 ) {

	    if ( distance <= fz_1st_70_cum ) {
	      quality = 5;
	    } else {
	      if ( distance <= fz_1st_50_cum ) {
	        quality = 4;
	      } else {
		quality = 3;
	      }
	    }

	  } else {

	    int firstStep = distances.at(0).first;
	    int firstDistance = distances.at(0).second;

            switch ( firstStep ) {

	      case 0:{

	        if ( ( firstDistance <= tw_1st_50 ) &&
	             ( distance <= tw_2nd_fz_cum ) ) {
	          quality = 2;
	        } else {
	          quality = 1;
	        }
	        break;

	      }

	      case 1:{

	        if ( ( ( firstDistance <= sc_1st_50 ) && ( distance <= sc_2nd_fz_0_cum  ) ) ||
	             ( ( firstDistance > sc_1st_50 )  && ( distance <= sc_2nd_fz_1_cum  ) ) ) {
	          quality = 2;
	        } else {
	          quality = 1;
	        }
	        break;

	      }

	      case 2:{

                if ( ( firstDistance <= fz_1st_50 ) &&
                     ( distance <= fz_2nd_fz_cum ) ) {
                  quality = 2;
                } else {
                  quality = 1;
                }
                break;

	      }

	      default:{
	        quality = 1;
	      }

            }

	  }

	} else {

	  // closest in 3D
	  if ( assoc_ite == 0 ) {

            if ( distance <= f3_1st_70_cum ) {
	      quality = 5;
            } else {
	      if ( distance <= f3_1st_50_cum ) {
	        quality = 4;
	      } else {
	        quality = 3;
	      }
	    }

	  } else {

	    int firstStep = distances.at(0).first;
	    int firstDistance = distances.at(0).second;

	    switch ( firstStep ) {

	      case 0:{

	        if ( ( firstDistance <= tw_1st_50 ) &&
	             ( distance <= tw_2nd_f3_cum ) ) {
	          quality = 2;
                } else {
	          quality = 1;
	        }
	        break;

	      }

	      case 1:{

	        if ( ( ( firstDistance <= sc_1st_50 ) && ( distance <= sc_2nd_f3_0_cum  ) ) ||
	             ( ( firstDistance > sc_1st_50 )  && ( distance <= sc_2nd_f3_1_cum  ) ) ) {
	          quality = 2;
	        } else {
	          quality = 1;
	        }
	        break;

	      }

	      case 2:{

	        if ( ( firstDistance <= fz_1st_50 ) &&
	             ( distance <= f3_2nd_f3_cum ) ) {
	          quality = 2;
	        } else {
	          quality = 1;
	        }
	        break;

	      }

	      default:{
	        quality = 1;
	      }

	    }

	  }
	}
      }

      break;
    }

    default:{
      quality = 0;
      break;
    }

  }

  return quality;

}
