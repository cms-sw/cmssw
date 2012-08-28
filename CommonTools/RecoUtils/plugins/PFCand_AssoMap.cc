// -*- C++ -*-
//
// Package:    PFCand_AssoMap
// Class:      PFCand_AssoMap
// 
/**\class PFCand_AssoMap PFCand_AssoMap.cc CommonTools/RecoUtils/plugins/PFCand_AssoMap.cc

  Description: Produces a map with association between pf candidates and their particular most probable vertex with a quality of this association
*/
//
// Original Author:  Matthias Geisler
//         Created:  Wed Apr 18 14:48:37 CEST 2012
// $Id: PFCand_AssoMap.cc,v 1.3 2012/06/06 09:02:22 mgeisler Exp $
//
//
#include "CommonTools/RecoUtils/interface/PFCand_AssoMap.h"

// system include files
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include <DataFormats/EgammaReco/interface/SuperCluster.h>
#include <DataFormats/EgammaReco/interface/SuperClusterFwd.h>
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

//
// constants, enums and typedefs
//
   
using namespace edm;
using namespace std;
using namespace reco;

typedef AssociationMap<OneToManyWithQuality< VertexCollection, PFCandidateCollection, float> > PFCandVertexAssMap;

typedef pair<PFCandidateRef, float> PFCandQualityPair;
typedef vector< PFCandQualityPair > PFCandQualityPairVector;

typedef pair<VertexRef, PFCandQualityPair> VertexPfcQuality;


typedef pair<TrackRef, float> TrackQualityPair;

typedef pair<VertexRef, TrackQualityPair> VertexTrackQuality;


//
// static data member definitions
//

//
// constructors and destructor
//
PFCand_AssoMap::PFCand_AssoMap(const edm::ParameterSet& iConfig):PF_PU_AssoMapAlgos(iConfig)
{
   //register your products

  	produces<PFCandVertexAssMap>();

   //now do what ever other initialization is needed

  	input_PFCandidates_ = iConfig.getParameter<InputTag>("PFCandidateCollection");

  
}


PFCand_AssoMap::~PFCand_AssoMap()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PFCand_AssoMap::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  	//std::cout << "<PFCand_AssoMap::produce>:" << std::endl;

	auto_ptr<PFCandVertexAssMap> pfCandAM(new PFCandVertexAssMap() );
  
	//get the input pfCandidateCollection
  	Handle<PFCandidateCollection> pfCandInH;
  	iEvent.getByLabel(input_PFCandidates_,pfCandInH);

   	if ( !PF_PU_AssoMapAlgos::GetInputCollections(iEvent,iSetup) ) return;

	VertexRef FirstVtxRef = PF_PU_AssoMapAlgos::GetFirstVertex();
   
	for( unsigned i=0; i<pfCandInH->size(); i++ ) {
     
          PFCandidateRef candref(pfCandInH, i);

	  float weight;

          VertexPfcQuality VtxPfcQual;

	  TrackRef PFCtrackref = candref->trackRef();

	  if ( PFCtrackref.isNull() ){
     
            //the pfcand has no reference to a general track, therefore its mostly uncharged
            //it will allways be associated to the first vertex,
            //this was found out to be the best solution w.r.t. jet-pt response
            //weight set to -3.

            weight = -3.;              
	    VtxPfcQual = make_pair(FirstVtxRef, make_pair(candref, weight));         

          } else {

    	    VertexTrackQuality VtxTrkQual = PF_PU_AssoMapAlgos::DoTrackAssociation(PFCtrackref, iSetup);
	    VtxPfcQual = make_pair(VtxTrkQual.first, make_pair(candref, VtxTrkQual.second.second)); 

	  }

    	  //std::cout << "associating candidate: Pt = " << VtxPfcQual.second.first->pt() << "," 
    	  //	        << " eta = " << VtxPfcQual.second.first->eta() << ", phi = " << VtxPfcQual.second.first->phi() 
    	  //	        << " to vertex: z = " << VtxPfcQual.first->position().z() << " with quality q = " << VtxPfcQual.second.second << std::endl;
    
    	  // Insert the best vertex and the pair of candidate and the quality of this association in the map
    	  pfCandAM->insert(VtxPfcQual.first, VtxPfcQual.second);

       	}

   	iEvent.put( PFCand_NoPU_WithAM_Algos::SortAssociationMap( &(*pfCandAM) ) );

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PFCand_AssoMap::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFCand_AssoMap);
