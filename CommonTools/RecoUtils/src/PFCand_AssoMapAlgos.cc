
#include "CommonTools/RecoUtils/interface/PFCand_AssoMapAlgos.h"

#include "TrackingTools/IPTools/interface/IPTools.h"

using namespace edm;
using namespace std;
using namespace reco;

/*************************************************************************************/
/* dedicated constructor for the algorithms                                          */ 
/*************************************************************************************/

PFCand_AssoMapAlgos::PFCand_AssoMapAlgos(const edm::ParameterSet& iConfig):PF_PU_AssoMapAlgos(iConfig)
{

  	input_MaxNumAssociations_ = iConfig.getParameter<int>("MaxNumberOfAssociations");

  	input_VertexCollection_= iConfig.getParameter<InputTag>("VertexCollection");

  	input_BeamSpot_= iConfig.getParameter<InputTag>("BeamSpot");

}

/*************************************************************************************/
/* get all needed collections at the beginning                                       */ 
/*************************************************************************************/

void 
PFCand_AssoMapAlgos::GetInputCollections(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	PF_PU_AssoMapAlgos::GetInputCollections(iEvent, iSetup);
  
  	//get the offline beam spot
  	iEvent.getByLabel(input_BeamSpot_, beamspotH);
	  
  	//get the input vertex collection
  	iEvent.getByLabel(input_VertexCollection_, vtxcollH);

     	iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);

}

/*************************************************************************************/
/* create the pf candidate to vertex association map                                 */ 
/*************************************************************************************/

std::auto_ptr<PFCandToVertexAssMap> 
PFCand_AssoMapAlgos::CreatePFCandToVertexMap(edm::Handle<reco::PFCandidateCollection> pfCandH, const edm::EventSetup& iSetup)
{

  	auto_ptr<PFCandToVertexAssMap> pfcand2vertex(new PFCandToVertexAssMap());

	int num_vertices = vtxcollH->size();
	if ( num_vertices < input_MaxNumAssociations_) input_MaxNumAssociations_ = num_vertices;
   
	for( unsigned i=0; i<pfCandH->size(); i++ ) {
     
          PFCandidateRef candref(pfCandH, i);

	  vector<VertexRef>* vtxColl_help = CreateVertexVector(vtxcollH);

          VertexPfcQuality VtxPfcQual;

	  TrackRef PFCtrackref = candref->trackRef();

	  if ( PFCtrackref.isNull() ){

	    for ( int assoc_ite = 0; assoc_ite < input_MaxNumAssociations_; ++assoc_ite ) {

	      int quality = -1 - assoc_ite; 

    	      // Insert the best vertex and the pair of track and the quality of this association in the map
    	      pfcand2vertex->insert( vtxColl_help->at(0), make_pair(candref, quality) );
 
	      PF_PU_AssoMapAlgos::EraseVertex(vtxColl_help, vtxColl_help->at(0));

	    }       

          } else {

            TransientTrack transtrk(PFCtrackref, &(*bFieldH) );
            transtrk.setBeamSpot(*beamspotH);
            transtrk.setES(iSetup);

	    for ( int assoc_ite = 0; assoc_ite < input_MaxNumAssociations_; ++assoc_ite ) {

    	      VertexStepPair assocVtx = FindAssociation(PFCtrackref, vtxColl_help, bFieldH, iSetup, beamspotH, assoc_ite);	 
	      int step = assocVtx.second;
	      double distance = ( IPTools::absoluteImpactParameter3D( transtrk, *(assocVtx.first) ) ).second.value();
	   
	      int quality = DefineQuality(assoc_ite, step, distance);
    
    	      // Insert the best vertex and the pair of track and the quality of this association in the map
    	      pfcand2vertex->insert( assocVtx.first, make_pair(candref, quality) );

	      PF_PU_AssoMapAlgos::EraseVertex(vtxColl_help, assocVtx.first);

	    }

	  }

       	}

	return pfcand2vertex;

}

/*************************************************************************************/
/* create the vertex to pf candidate association map                                 */ 
/*************************************************************************************/

std::auto_ptr<VertexToPFCandAssMap> 
PFCand_AssoMapAlgos::CreateVertexToPFCandMap(edm::Handle<reco::PFCandidateCollection> pfCandH, const edm::EventSetup& iSetup)
{

  	auto_ptr<VertexToPFCandAssMap> vertex2pfcand(new VertexToPFCandAssMap());

	int num_vertices = vtxcollH->size();
	if ( num_vertices < input_MaxNumAssociations_) input_MaxNumAssociations_ = num_vertices;
   
	for( unsigned i=0; i<pfCandH->size(); i++ ) {
     
          PFCandidateRef candref(pfCandH, i);

	  vector<VertexRef>* vtxColl_help = CreateVertexVector(vtxcollH);

          VertexPfcQuality VtxPfcQual;

	  TrackRef PFCtrackref = candref->trackRef();

	  if ( PFCtrackref.isNull() ){

	    for ( int assoc_ite = 0; assoc_ite < input_MaxNumAssociations_; ++assoc_ite ) {

	      int quality = -1 - assoc_ite; 

    	      // Insert the best vertex and the pair of track and the quality of this association in the map
    	      vertex2pfcand->insert( candref, make_pair(vtxColl_help->at(0), quality) );
 
	      PF_PU_AssoMapAlgos::EraseVertex(vtxColl_help, vtxColl_help->at(0));

	    }       

          } else {

            TransientTrack transtrk(PFCtrackref, &(*bFieldH) );
            transtrk.setBeamSpot(*beamspotH);
            transtrk.setES(iSetup);

	    for ( int assoc_ite = 0; assoc_ite < input_MaxNumAssociations_; ++assoc_ite ) {

    	      VertexStepPair assocVtx = FindAssociation(PFCtrackref, vtxColl_help, bFieldH, iSetup, beamspotH, assoc_ite);	 
	      int step = assocVtx.second;
	      double distance = ( IPTools::absoluteImpactParameter3D( transtrk, *(assocVtx.first) ) ).second.value();
	   
	      int quality = DefineQuality(assoc_ite, step, distance);
    
    	      // Insert the best vertex and the pair of track and the quality of this association in the map
    	      vertex2pfcand->insert( candref, make_pair(assocVtx.first, quality) );

	      PF_PU_AssoMapAlgos::EraseVertex(vtxColl_help, assocVtx.first);

	    }

	  }

       	}

	return vertex2pfcand;
}

/*************************************************************************************/
/* create the vertex to pf candidate association map                                 */ 
/*************************************************************************************/

std::auto_ptr<PFCandToVertexAssMap> 
PFCand_AssoMapAlgos::SortPFCandAssociationMap(PFCandToVertexAssMap* pfcvertexassInput)
{
	//create a new PFCandVertexAssMap for the Output which will be sorted
     	auto_ptr<PFCandToVertexAssMap> pfcvertexassOutput(new PFCandToVertexAssMap() );

	//Create and fill a vector of pairs of vertex and the summed (pT)**2 of the pfcandidates associated to the vertex 
	VertexPtsumVector vertexptsumvector;

	//loop over all vertices in the association map
        for(PFCandToVertexAssMap::const_iterator assomap_ite=pfcvertexassInput->begin(); assomap_ite!=pfcvertexassInput->end(); assomap_ite++){

	  const VertexRef assomap_vertexref = assomap_ite->key;
  	  const PFCandQualityPairVector pfccoll = assomap_ite->val;

	  float ptsum = 0;
 
	  PFCandidateRef pfcandref;

	  //get the pfcandidates associated to the vertex and calculate the pT**2
	  for(unsigned int pfccoll_ite=0; pfccoll_ite<pfccoll.size(); pfccoll_ite++){

	    pfcandref = pfccoll[pfccoll_ite].first;
	    int quality = pfccoll[pfccoll_ite].second;

	    if ( (quality<=2) && (quality!=-1) ) continue; 	 

	    double man_pT = pfcandref->pt();
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
          for(PFCandToVertexAssMap::const_iterator assomap_ite=pfcvertexassInput->begin(); assomap_ite!=pfcvertexassInput->end(); assomap_ite++){

	    const VertexRef assomap_vertexref = assomap_ite->key;
  	    const PFCandQualityPairVector pfccoll = assomap_ite->val;

	    //if the vertex from the association map the vertex with the highest pT 
	    //insert all associated pfcandidates in the output Association Map
	    if(assomap_vertexref==vertexref_highestpT) 
	      for(unsigned int pfccoll_ite=0; pfccoll_ite<pfccoll.size(); pfccoll_ite++) 
	        pfcvertexassOutput->insert(assomap_vertexref,pfccoll[pfccoll_ite]);
 
	  }

	  vertexptsumvector.erase(vertexptsumvector.begin()+highestpT_index);	

	}

  	return pfcvertexassOutput;
}