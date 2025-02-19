// -*- C++ -*-
//
// Package:    PF_PU_AssoMap
// Class:      PF_PU_AssoMap
// 
/**\class PF_PU_AssoMap PF_PU_AssoMap.cc CommonTools/RecoUtils/plugins/PF_PU_AssoMap.cc

 Description: Produces a map with association between tracks and their particular most probable vertex with a quality of this association

*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//

#include "CommonTools/RecoUtils/interface/PF_PU_AssoMap.h"

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

#include "TMath.h"
   
using namespace edm;
using namespace std;
using namespace reco;

typedef AssociationMap<OneToManyWithQuality<VertexCollection, TrackCollection, float> > TrackVertexAssMap;

typedef pair<TrackRef, float> TrackQualityPair;
typedef vector<TrackQualityPair > TrackQualityPairVector;

typedef pair<VertexRef, TrackQualityPair> VertexTrackQuality;

typedef vector<VertexRef > VertexRefV;

//
// constructors and destructor
//
PF_PU_AssoMap::PF_PU_AssoMap(const edm::ParameterSet& iConfig):PF_PU_AssoMapAlgos(iConfig)
{ 
   //register your products

  	produces<TrackVertexAssMap>();

   //now do what ever other initialization is needed

  	input_TrackCollection_ = iConfig.getParameter<InputTag>("TrackCollection");
  
}


PF_PU_AssoMap::~PF_PU_AssoMap()
{
// nothing to be done yet...
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
PF_PU_AssoMap::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  	//std::cout << "<PF_PU_AssoMap::produce>:" << std::endl;

  	auto_ptr<TrackVertexAssMap> trackvertexass(new TrackVertexAssMap());
    
  	//get the input track collection     
  	Handle<TrackCollection> trkcollH;
  	iEvent.getByLabel(input_TrackCollection_, trkcollH);

  	if ( !PF_PU_AssoMapAlgos::GetInputCollections(iEvent,iSetup) ){

  	  iEvent.put( trackvertexass );
          return;

        }
	    
  	//loop over all tracks of the track collection	
  	for ( size_t idxTrack = 0; idxTrack < trkcollH->size(); ++idxTrack ) {

    	  TrackRef trackref = TrackRef(trkcollH, idxTrack);

    	  VertexTrackQuality VtxTrkQual_tmp = PF_PU_AssoMapAlgos::DoTrackAssociation(trackref, iSetup);

    	  //std::cout << "associating track: Pt = " << VtxTrkQual_tmp.second.first->pt() << "," 
    	  //	        << " eta = " << VtxTrkQual_tmp.second.first->eta() << ", phi = " << VtxTrkQual_tmp.second.first->phi() 
    	  //	        << " to vertex: z = " << VtxTrkQual_tmp.first->position().z() << " with quality q = " << VtxTrkQual_tmp.second.second << std::endl;
    
    	  // Insert the best vertex and the pair of track and the quality of this association in the map
    	  trackvertexass->insert(VtxTrkQual_tmp.first, VtxTrkQual_tmp.second);

  	}

  	iEvent.put( PF_PU_AssoMapAlgos::SortAssociationMap( &(*trackvertexass) ) );
	
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PF_PU_AssoMap::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  // The following says we do not know what parameters are allowed so do no validation.
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PF_PU_AssoMap);
